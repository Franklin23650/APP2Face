using Microsoft.AspNetCore.Cors.Infrastructure;
using Microsoft.ML.OnnxRuntime;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllers();

// Habilita Swagger/OpenAPI
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

string corsPolicy = "CorsPolicy";

// Configura CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy(corsPolicy,
    builder => builder
        .AllowAnyOrigin()
        .AllowAnyHeader()
        .AllowAnyMethod()
        .Build());
});



builder.Services.AddSingleton<InferenceSession>(sp =>
{
    var modelPath = Path.Combine(AppContext.BaseDirectory, "Models/arcface.onnx");
    if (!System.IO.File.Exists(modelPath))
        throw new InvalidOperationException("El modelo no se encontró en la ruta especificada.");
    return new InferenceSession(modelPath);
});

var app = builder.Build();

// Habilita Swagger en desarrollo
//if (app.Environment.IsDevelopment() || app.Environment.IsStaging() || app.Environment.IsProduction())
//{
app.UseSwagger();
app.UseSwaggerUI();
//}

// Usa la política de CORS antes de MapControllers
app.UseCors(corsPolicy);

app.UseHttpsRedirection();

app.MapControllers();

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.MapGet("/weatherforecast", () =>
{
    var forecast = Enumerable.Range(1, 5).Select(index =>
        new WeatherForecast
        (
            DateOnly.FromDateTime(DateTime.Now.AddDays(index)),
            Random.Shared.Next(-20, 55),
            summaries[Random.Shared.Next(summaries.Length)]
        ))
        .ToArray();
    return forecast;
})
.WithName("GetWeatherForecast");

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}