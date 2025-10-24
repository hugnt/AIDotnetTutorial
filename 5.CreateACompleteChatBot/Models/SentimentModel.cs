using Microsoft.ML.Data;

public class SentimentData
{
    // LoadColumn(0): Chỉ định rằng property này sẽ được load từ cột 0 của file dữ liệu
    [LoadColumn(0)]
    public string Text { get; set; } = string.Empty;

    // LoadColumn(1): Load từ cột 1 của file dữ liệu
    // ColumnName("Label"): Đặt tên cột trong ML pipeline là "Label"
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; } // true = positive, false = negative
}

public class SentimentPrediction
{
    // ColumnName("PredictedLabel"): Map với cột "PredictedLabel" từ output của mô hình
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; } // Kết quả dự đoán: true = positive, false = negative

    // Xác suất của prediction (từ 0.0 đến 1.0)
    public float Probability { get; set; }

    // Điểm số raw từ mô hình (có thể âm hoặc dương)
    public float Score { get; set; }
}