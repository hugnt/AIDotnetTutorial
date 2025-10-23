using Microsoft.ML.Data;

public class ReviewInput
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public bool Label { get; set; }
}

public class ReviewPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
}
