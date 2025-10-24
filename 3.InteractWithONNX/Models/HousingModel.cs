using Microsoft.ML.Data;

public class HousingPrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}

public class HousingData
{
    [LoadColumn(0)]
    [ColumnName("Size")]
    public float Size { get; set; }

    [LoadColumn(1, 3)]
    [VectorType(3)]
    [ColumnName("HistoricalPrices")]
    public float[] HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}

public class HousingInput
{
    public float Size { get; set; } = 120f;
    public float[] HistoricalPrices { get; set; } = new float[] { 100000f, 105000f, 110000f };
}


public class HousingOnnxInput
{
    [ColumnName("Size")]
    public float Size { get; set; }

    [ColumnName("HistoricalPrices")]
    public float[] HistoricalPrices { get; set; }
}

public class HousingOnnxOutput
{
    [ColumnName("Score.output")]
    public float PredictedPrice { get; set; }
}
