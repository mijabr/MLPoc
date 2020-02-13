using System;
using System.Collections.Generic;
using System.Linq;

namespace ML
{
    public class RandomDataGenerator
    {
        public IEnumerable<ModelInput> GetRandomData()
        {
            var r = new Random();

            var dates = Enumerable.Range(0, 1095).Select(i => DateTime.Parse("2015-01-01").AddDays(i));

            int lastVal = r.Next(20000, 50000);
            return dates.Select(d =>
            {
                int nextVal = Math.Max(0, lastVal + r.Next(-5000, 5000));
                lastVal = nextVal;
                return new ModelInput()
                {
                    Date = d,
                    YearInSeries = d.Year - 2015,
                    TotalSold = nextVal
                };
            });
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
    }
}
