using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace AIDotnetTutorial.Services;

public interface IHousingPredictionService
{
    void TrainAndSaveModel();
    HousingOnnxOutput Predict(HousingInput inputData);
}

public class HousingPredictionService : IHousingPredictionService
{
    public HousingPredictionService()
    {

    }

    public void TrainAndSaveModel()
    {
        HousingData[] housingData = new HousingData[]
        {
            new HousingData
            {
                Size = 600f,
                HistoricalPrices = new float[] { 100000f, 125000f, 122000f },
                CurrentPrice = 170000f
            },
            new HousingData
            {
                Size = 1000f,
                HistoricalPrices = new float[] { 200000f, 250000f, 230000f },
                CurrentPrice = 225000f
            },
            new HousingData
            {
                Size = 1000f,
                HistoricalPrices = new float[] { 126000f, 130000f, 200000f },
                CurrentPrice = 195000f
            }
        };

        // Create MLContext
        MLContext mlContext = new MLContext();

        // Load Data
        IDataView data = mlContext.Data.LoadFromEnumerable<HousingData>(housingData);

        // Define data preparation estimator
        EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> pipelineEstimator =
            mlContext.Transforms.Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Regression.Trainers.Sdca());

        // Train the model
        var model = pipelineEstimator.Fit(data);

        //Save the model
        using (var stream = File.Create("./ExportedModels/onnx_model.onnx"))
        {
            mlContext.Model.ConvertToOnnx(
                transform: model,
                inputData: data,
                stream: stream,
                outputColumns: new[] { "Score" }
            );
        }
    }


    //https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automl-onnx-model-dotnet?view=azureml-api-2&toc=%2Fdotnet%2Fmachine-learning%2Ftoc.json&bc=%2Fdotnet%2Fmachine-learning%2Ftoc.json
    public HousingOnnxOutput Predict(HousingInput inputData)
    {
        //Create MLContext
        var mlContext = new MLContext();
        var transformer = BuildOnnxTransformer(mlContext, "./ExportedModels/onnx_model.onnx");
        var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingOnnxInput, HousingOnnxOutput>(transformer);
        // Load Trained Model
        var onnxInput = new HousingOnnxInput
        {
            Size = inputData.Size,
            HistoricalPrices = inputData.HistoricalPrices
        };


        var prediction = onnxPredictionEngine.Predict(onnxInput);
        Console.WriteLine($"Input Size={onnxInput.Size}, Hist=[{string.Join(",", onnxInput.HistoricalPrices)}] => Predicted={prediction.PredictedPrice}");
        return prediction;
    }

    // 3) Build pipeline with explicit mapping + inspect schema & values
    public static ITransformer BuildOnnxTransformer(MLContext ml, string onnxPath)
    {
        string[] inputNames = new[] { "Size", "HistoricalPrices" };
        string[] outputNames = new[] { "Score.output" };

        var est = ml.Transforms.ApplyOnnxModel(
            modelFile: onnxPath,
            outputColumnNames: outputNames,
            inputColumnNames: inputNames
        );

        var empty = ml.Data.LoadFromEnumerable(new HousingData[] { }); // empty schema
        var transformer = est.Fit(empty);

        // Debug: in schema created by transformer
        var dv = transformer.Transform(empty);
        Console.WriteLine("Transformed schema:");
        foreach (var c in dv.Schema) Console.WriteLine($"  {c.Index}: {c.Name} -> {c.Type}");

        return transformer;
    }

}