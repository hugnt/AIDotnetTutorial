using Microsoft.ML;

//1. Khởi tạo MLContext
var mlContext = new MLContext(seed: 42);

//2. Load dữ liệu mẫu
var trainPath = Path.Combine("Data", "train.tsv");
var testPath = Path.Combine("Data", "test.tsv");

var trainDataView = mlContext.Data.LoadFromTextFile<ReviewInput>(trainPath, hasHeader: true, separatorChar: '\t');
var testDataView = mlContext.Data.LoadFromTextFile<ReviewInput>(testPath, hasHeader: true, separatorChar: '\t');

//2b. using trainTestSplit 
// var dataPath = Path.Combine("Data", "train.tsv");
// var data = mlContext.Data.LoadFromTextFile<ReviewInput>(dataPath, hasHeader: true, separatorChar: '\t');
// var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, samplingKeyColumnName: nameof(ReviewInput.Label), seed: 123);
// var trainSet = split.TrainSet;
// var testSet  = split.TestSet;

//3. Xây dựng Pipeline
var pipeline = mlContext.Transforms.Text
    // FeaturizeText: chuyển chuỗi text thành vector số (n-grams, TF-IDF, bag-of-words...). 
    // -> Tạo cột đầu ra "TextFeaturized" chứa các đặc trưng số cho văn bản.
    .FeaturizeText(outputColumnName: "TextFeaturized", inputColumnName: nameof(ReviewInput.Text))

    // NormalizeMinMax: chuẩn hóa các giá trị trong "TextFeaturized" về khoảng [0,1].
    // -> Giúp các thuật toán (như logistic regression, SVM, ...) hội tụ ổn định hơn,
    //    và tránh rằng các chiều có magnitude lớn áp đảo các chiều khác.
    .Append(mlContext.Transforms.NormalizeMinMax("TextFeaturized"))

    // Concatenate: gộp một hoặc nhiều cột feature vào cột duy nhất "Features". (định dạng mà hầu hết trainer yêu cầu)
    // -> Các trainers của ML.NET mặc định nhận input là cột "Features" (vector duy nhất).
    //    Ở đây chúng ta chỉ gộp "TextFeaturized" (có thể thêm cột numeric khác nếu có).
    .Append(mlContext.Transforms.Concatenate("Features", "TextFeaturized"))

    // Trainer: sdca logistic regression cho binary classification.
    // - labelColumnName: tên cột chứa nhãn (ở đây là "Label").
    // - featureColumnName: tên cột features (ở đây dùng "Features" vừa tạo).
    // -> Kết quả là một estimator/transformer huấn luyện mô hình phân loại nhị phân.
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: "Label", featureColumnName: "Features"));

// 4. Huấn luyện mô hình
// Fit là lệnh kích hoạt huấn luyện/“fit” của một Estimator (ở đây là pipeline) trên dữ liệu (trainDataView).
// Fit biến chuỗi các bước mô tả (Estimator pipeline) thành một pipeline thực thi (Transformer) dùng để dự đoán.
var model = pipeline.Fit(trainDataView);
Console.WriteLine("Model training completed.");

// 5. Đánh giá mô hình trên test set
// Dùng model đã huấn luyện để chuyển testDataView thành tập dự đoán (scored data)
var predictions = model.Transform(testDataView);

// Đánh giá bộ dự đoán bằng hàm Evaluate của MLContext.
// - predictions: IDataView sau khi đã có các cột do model sinh (Score, Probability, PredictedLabel...).
// - labelColumnName: tên cột chứa nhãn thực tế trong test set (ở ví dụ này là "Label").
var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");
Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");

// 6. Tạo Prediction Engine (single prediction)
var engine = mlContext.Model.CreatePredictionEngine<ReviewInput, ReviewPrediction>(model);
Console.WriteLine("Prediction engine created.");

// 7. Dự đoán mẫu
var sample = new ReviewInput { Text = "Quá tuyệt vời, dịch vụ rất chuyên nghiệp" };
var pred = engine.Predict(sample);
Console.WriteLine($"Prediction: {pred.Prediction}, Probability: {pred.Probability:P2}, Score: {pred.Score}");

//8. Lưu mô hình
var modelPath = Path.Combine("ExportedModels", "reviewModel.zip");
mlContext.Model.Save(model, trainDataView.Schema, modelPath);
Console.WriteLine($"Model saved to {modelPath}");