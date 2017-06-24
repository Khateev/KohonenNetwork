using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KohonenNetwork
{
    public class KohonenNetwork
    {
        public Neuron[] Clusters;

        private double _trainVelocity = 0.3;
        private double _potentialMin = 0.4; //0.75
        private int _lengthDataVector;
        private Normalize.Normalize _normalize;

        public KohonenNetwork(int amountClusters, List<double[]> data)
        {
            _normalize = new Normalize.Normalize(data);
            Clusters = new Neuron[amountClusters];
            InitClustersWeights(data[0].Length, _potentialMin);
        }

        private void InitClustersWeights(int dim_input_vector, double potential)
        {
            var random = new Random();
            for (var i = 0; i < Clusters.Length; i++)
            {
                var weights = new double[dim_input_vector];
                for (var j = 0; j < dim_input_vector; j++)
                {
                    weights[j] = random.NextDouble();
                }
                Clusters[i] = new Neuron(weights, potential);
            }
        }

        public void Train(List<double[]> input)
        {
            var normalizationData = input.Select(x => _normalize.Normalization(x)).ToArray();

            var edge = 0;
            for (var v = _trainVelocity; v > 0; v = v - 0.01)
            {
                for (var i = 0; i < input.Count(); i++)
                {
                    var numberTrain = edge * (normalizationData[i].Length - 1) + i + 1;

                    var winner = GetWinnerInRace(normalizationData[i]);
                    ChangePotentialAllNeurons(winner);

                    winner.CorrectWinner(normalizationData[i], v);
                    CorrectNeighborsOfWinner(winner, normalizationData[i], v, edge);

                }

                edge++;
            }
        }

        public int[] ToLabel(List<double[]> input)
        {
            return input.Select(x => GetIndexNearestToVector(Clusters, _normalize.Normalization(x))).ToArray();
        }

        private Neuron[] GetNeuronsWithCorrectPotential()
        {
            return Clusters.Where(x => x.GetPotential() >= _potentialMin).ToArray();
        }

        private Neuron GetWinnerInRace(double[] vector)
        {
            var neuronsInRace = GetNeuronsWithCorrectPotential();

            var winnerIndex = GetIndexNearestToVector(neuronsInRace, vector);
            return neuronsInRace[winnerIndex];
        }

        private int GetIndexNearestToVector(Neuron[] neurons, double[] vector)
        {
            var distances = neurons.Select(x => x.GetDistanceToVector(vector)).ToArray();
            return Array.IndexOf(distances, distances.Min());
        }

        private void ChangePotentialAllNeurons(Neuron winner)
        {
            foreach (var neuron in Clusters)
            {
                neuron.ChangePotential(neuron == winner, _potentialMin, 1.0 / Clusters.Length);
            }
        }

        private void CorrectNeighborsOfWinner(Neuron winner, double[] vector, double velocity, int edge)
        {
            var neighbors = Clusters.Where(x => x != winner).ToArray();
            foreach (var neighbor in neighbors)
            {
                neighbor.CorrectNeighbor(winner.GetWeights(), vector, velocity, edge);
            }
        }
    }




    public class Neuron
    {
        public double[] _weights { get; }
        private double _potential;
        private int _amountWinner = 0;

        public Neuron(double[] weights, double potential)
        {
            _weights = weights;
            _potential = potential;
        }

        public double GetDistanceToVector(double[] vector)
        {
            double distance = 0.0;
            for (var i = 0; i < _weights.Length; i++)
            {
                distance += Math.Pow(vector[i] - _weights[i], 2);
            }
            return Math.Sqrt(distance);
        }

        public double GetPotential()
        {
            return _potential;
        }

        public double[] GetWeights()
        {
            return _weights;
        }

        public void ChangePotential(bool isWinner, double potentialMin, double diffPotential)
        {
            if (isWinner)
            {
                _amountWinner++;
                _potential -= potentialMin;
                return;
            }
            _potential += diffPotential;       /////////change  ddd
        }

        public void CorrectWinner(double[] vector, double velocityTrain)
        {
            for (var i = 0; i < _weights.Length; i++)
            {
                _weights[i] += velocityTrain * (vector[i] - _weights[i]);
            }
        }

        public void CorrectNeighbor(double[] winnerWeights, double[] vector, double velocityTrain, double edge)
        {
            var sigma = Math.Pow(0.1 * Math.Exp(-1 * edge / 1000), 2);
            var functionOfNeighbors = Math.Exp(-1 * GetDistanceToVector(winnerWeights) / (2 * sigma));
            for (var i = 0; i < _weights.Length; i++)
            {
                _weights[i] += velocityTrain * functionOfNeighbors * (vector[i] - _weights[i]);
            }
        }

    }

    public interface IDistanceMeasure
    {
        double Distance(double[] vector1, double[] vector2);
    }

    public class EuclideanDistance : IDistanceMeasure
    {
        public double Distance(double[] vector1, double[] vector2)
        {
            double distance = 0.0;
            for (var i = 0; i < vector1.Length; i++)
            {
                distance += Math.Pow(vector1[i] - vector2[i], 2);
            }
            return Math.Sqrt(distance);
        }
    }

    public class ManhattanDistance : IDistanceMeasure
    {
        public double Distance(double[] vector1, double[] vector2)
        {
            double distance = 0.0;
            for (var i = 0; i < vector1.Length; i++)
            {
                distance += Math.Abs(vector1[i] - vector2[i]);
            }
            return distance;
        }
    }

    public class PowerDistance : IDistanceMeasure
    {
        private double _r;
        private double _p;

        public PowerDistance(double r, double p)
        {
            _r = r;
            _p = p;
        }

        public double Distance(double[] vector1, double[] vector2)
        {
            double distance = 0.0;
            for (var i = 0; i < vector1.Length; i++)
            {
                distance += Math.Pow(vector1[i] - vector2[i], _p);
            }
            return Math.Pow(distance, 1 / _r);
        }
    }
}
