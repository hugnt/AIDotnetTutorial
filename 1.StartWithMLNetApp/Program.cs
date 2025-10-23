using Microsoft.ML;

//1. Tạo một instance của MLContext - đây là điểm bắt đầu cho mọi thao tác ML.NET
var mlContext = new MLContext();

//2. Load dữ liệu training
var data = mlContext.Data.LoadFromTextFile<SentimentData>("Data/data.tsv", hasHeader: true);

//3. Tạo Machine Learning Pipeline
var pipeline = mlContext.Transforms.Text
    // Bước 1: FeaturizeText - Chuyển đổi text thành vector số
    // "Features": Tên cột output chứa features
    // nameof(SentimentData.Text): Sử dụng cột Text từ SentimentData làm input
    .FeaturizeText("Features", nameof(SentimentData.Text))

    // Bước 2: Append thuật toán SdcaLogisticRegression
    // Đây là thuật toán binary classification phù hợp cho sentiment analysis
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

//4. Train mô hình
var model = pipeline.Fit(data);

//5. Tạo Prediction Engine
// CreatePredictionEngine: Tạo engine để thực hiện prediction
// <SentimentData, SentimentPrediction>: Chỉ định kiểu input và output
var engine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

//6. Test mô hình với dữ liệu mẫu
var sample = new SentimentData { Text = "Total waste of time and money" };

//7. Predict: Thực hiện dự đoán cảm xúc cho sample
var prediction = engine.Predict(sample);

Console.WriteLine($"Positive: {prediction.Prediction}, Score: {prediction.Score}");