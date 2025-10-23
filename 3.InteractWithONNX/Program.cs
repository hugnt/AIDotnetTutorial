using AIDotnetTutorial.Services;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddOpenApi();
builder.Services.AddSwaggerGen();

// DI Services
builder.Services.AddSingleton<IHousingPredictionService, HousingPredictionService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();


app.MapGet("/train-model", (IHousingPredictionService housingPredictionService) =>
{
    housingPredictionService.TrainAndSaveModel();
    return Results.Ok("Model trained and saved.");
})
.WithName("TrainModel");

app.MapPost("/predict", (IHousingPredictionService housingPredictionService, [FromBody] HousingInput inputData) =>
{
    var prediction = housingPredictionService.Predict(inputData);
    return Results.Ok(prediction);
});

app.Run();
