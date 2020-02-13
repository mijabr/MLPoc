using Microsoft.ML;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Transforms.TimeSeries;

namespace ML
{
    public class Program
    {
        public static string DataFileName => "data.json";
        public static string ModelFileName => "model.zip";

        public static void Main(string[] args)
        {
            MakeRandomData();
            RunModel();
        }

        private static void MakeRandomData()
        {
            var data = new RandomDataGenerator().GetRandomData();
            File.WriteAllText(DataFileName, JsonConvert.SerializeObject(data));
        }

        private static void RunModel()
        {
            var modelRunner = new ModelRunner();

            var dataSource = JsonConvert.DeserializeObject<List<ModelRunner.ModelInput>>(File.ReadAllText(DataFileName));

            string modelPath = Path.Combine(ModelFileName);

            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromEnumerable(dataSource);

            IDataView AllData = mlContext.Data.FilterRowsByColumn(dataView, "YearInSeries", 0, 3);
            IDataView Year2017Data = mlContext.Data.FilterRowsByColumn(dataView, "YearInSeries", 2, 3);

            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedTotalSold",
                inputColumnName: "TotalSold",
                windowSize: 30,
                seriesLength: 365,
                trainSize: 1095,
                horizon: 365,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundTotalSold",
                confidenceUpperBoundColumn: "UpperBoundTotalSold");


            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(AllData);
            modelRunner.Evaluate(Year2017Data, forecaster, mlContext);

            var forecastingEngine = forecaster.CreateTimeSeriesEngine<ModelRunner.ModelInput, ModelRunner.ModelOutput>(mlContext);
            forecastingEngine.CheckPoint(mlContext, modelPath);

            modelRunner.Forecast(Year2017Data, 365, forecastingEngine, mlContext);
        }
	}
}
