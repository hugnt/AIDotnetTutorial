using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddKernel();
builder.Services.AddAzureOpenAIChatCompletion(
    deploymentName: builder.Configuration["OpenAI:DeploymentId"],
    endpoint: builder.Configuration["OpenAI:Endpoint"],
    apiKey: builder.Configuration["OpenAI:ApiKey"]
);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.MapGet("/weatherforecast", () =>
{
    return "sss";
})
.WithName("GetWeatherForecast");

app.Run();