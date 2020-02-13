using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML
{
	public class ModelRunner
    {
		public void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
		{
			IDataView predictions = model.Transform(testData);

			IEnumerable<float> actual = mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
				.Select(observed => observed.TotalSold);

			IEnumerable<float> forecast = mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
				.Select(prediction => prediction.ForecastedTotalSold[0]);

			var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

			var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
			var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

			Console.WriteLine("Evaluation Metrics");
			Console.WriteLine("---------------------");
			Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
			Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
		}

		public void Forecast(IDataView compareAgainst, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
		{
			ModelOutput forecast = forecaster.Predict(horizon);

			IEnumerable<string> forecastOutput =
			mlContext.Data.CreateEnumerable<ModelInput>(compareAgainst, reuseRowObject: false)
				.Take(horizon)
				.Select((ModelInput input, int index) =>
				{
					string purchaseDate = input.Date.ToString("yyyy-MM-dd");
					float actualSales = input.TotalSold;
					float lowerEstimate = Math.Max(0, forecast.LowerBoundTotalSold[index]);
					float estimate = forecast.ForecastedTotalSold[index];
					float upperEstimate = forecast.UpperBoundTotalSold[index];
					return $"Date: {purchaseDate}\n" +
					$"Actual Sales: {actualSales}\n" +
					$"Lower Estimate: {lowerEstimate}\n" +
					$"Forecast: {estimate}\n" +
					$"Upper Estimate: {upperEstimate}\n";
				});

			Console.WriteLine("Purchase Forecast");
			Console.WriteLine("---------------------");
			foreach (var prediction in forecastOutput)
			{
				Console.WriteLine(prediction);
			}
		}

		public class ModelInput
		{
			public DateTime Date { get; set; }

			public float YearInSeries { get; set; }

			public float TotalSold { get; set; }

			public ModelInput() { }

			public ModelInput(string date, int yearInSeries, int totalSold)
			{
				this.Date = DateTime.Parse(date);
				this.YearInSeries = yearInSeries;
				this.TotalSold = totalSold;
			}
		}

		public class ModelOutput
		{
			public float[] ForecastedTotalSold { get; set; }
			public float[] LowerBoundTotalSold { get; set; }
			public float[] UpperBoundTotalSold { get; set; }
		}
	}
}
