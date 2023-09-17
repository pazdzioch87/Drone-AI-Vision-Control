using ControlBroker.MotionControl;
using Movement;

var builder = WebApplication.CreateBuilder(args);
builder
  .WebHost
  .UseKestrel()
  //.UseUrls("https://0.0.0.0:44353")
  .UseUrls("http://0.0.0.0:8001")
//    .UseIISIntegration()
  ;

builder.Services.AddCors(options =>
{
    options.AddPolicy("CorsPolicy", builder => builder
        //.WithOrigins(
        //"http://localhost:4200",
        //"https://localhost:4200",
        //"https://192.168.1.13:4200",
        //"http://192.168.1.13:4200",
        //"http://192.168.1.10:4200",
        //"https://192.168.1.10:4200",
        //"https://dronecontrolcenter.azurewebsites.net",
        //"http://dronecontrolcenter.azurewebsites.net",
        //"https://controlcenter.droneplatform.eu",
        //"http://controlcenter.droneplatform.eu"
        //)
        .SetIsOriginAllowed(x => _ = true)
        .SetIsOriginAllowedToAllowWildcardSubdomains()
        .AllowAnyOrigin()
        .AllowAnyMethod()
        .AllowAnyHeader()
        //.AllowCredentials()
        );
});


builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSignalR();
//builder.Services.AddSingleton<IMovementChannel, MovementChannel>();

var app = builder.Build();

//app.UseHttpsRedirection();
if (app.Environment.IsDevelopment())
{
    //app.UseSwagger();
    //app.UseSwaggerUI();
}

app.UseSwagger();
app.UseSwaggerUI();

// // Configure the HTTP request pipeline.
var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.MapGet("/weatherforecast", () =>
{
    var forecast = Enumerable.Range(1, 5).Select(index =>
        new WeatherForecast
        (
            DateTime.Now.AddDays(index),
            Random.Shared.Next(-20, 55),
            summaries[Random.Shared.Next(summaries.Length)]
        ))
        .ToArray();
    return forecast;
})
.WithName("GetWeatherForecast");

app.MapHub<MovementHub>("/movementHub");

//app.Use(async (context, next) =>
//{
//    context.Response.OnStarting(() =>
//    {
//        if (!context.Response.Headers.ContainsKey("Access-Control-Allow-Credentials"))
//        {
//            context.Response.Headers.Add("Access-Control-Allow-Credentials", "true");
//        }
//        if (!context.Response.Headers.ContainsKey("Access-Control-Allow-Origin"))
//        {
//            context.Response.Headers.Add("Access-Control-Allow-Origin", "*");
//        }
//        return Task.CompletedTask;
//    });

//    await next(context);
//    //return async context =>
//    //{
//    //};
//});

//app.Run(async context =>
//{
//    await context.Response.WriteAsync("Hello world!");
//});

app.UseCors("CorsPolicy");

app.Run();
internal record WeatherForecast(DateTime Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}
