using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Normalize
{
    public class Normalize
    {
        private double[] _max;
        private double[] _min;

        public Normalize(List<double[]> data)
        {
            _max = new double[data[0].Length];
            _min = new double[data[0].Length];
            for (var i = 0; i < data[0].Length; i++)
            {
                _max[i] = data.Max(x => x[i]);
                _min[i] = data.Min(x => x[i]);
            }
        }

        public double[] Normalization(double[] data)
        {
            return data.Select((z, i) => (z - _min[i]) / (_max[i] - _min[i])).ToArray();
        }

        public double[] Denormalization(double[] data)
        {
            return data.Select((x, i) => _min[i] + x * (_max[i] - _min[i])).ToArray();
        }
    }
}
