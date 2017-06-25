using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KohonenNetwork
{
    public class KohonenNetwork
    {
        public Neuron[] Clusters { get; private set; }

        private double _trainVelocity = 0.3;
        private double _potentialMin = 0.4; //0.75

        private IDistanceMeasure _distanceMeasure;

        public int Dimensions { get; private set; }
        public int AmountClusters { get; private set; }

        public KohonenNetwork(int amountClusters, int dimensions, IDistanceMeasure distanceMeasure = null, List<double[]> weights = null)
        {
            if (distanceMeasure == null)
            {
                _distanceMeasure = new EuclideanDistance();
            }
            else
            {
                _distanceMeasure = distanceMeasure;
            }

            Clusters = new Neuron[amountClusters];
            if (weights == null)
            {
                InitClustersWeights(dimensions, _potentialMin);
            }
            else
            {
                InitClustersWeights(weights, dimensions, _potentialMin);
            }
            Dimensions = dimensions;
            AmountClusters = amountClusters;

        }

        private void InitClustersWeights(List<double[]> weights, int dim_input_vector, double potential)
        {
            if (weights.Count != dim_input_vector)
            {
                //exception
            }

            for (var i = 0; i < Clusters.Length; i++)
            {
                Clusters[i] = new Neuron(weights[i], potential, _distanceMeasure, i);
            }
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
                Clusters[i] = new Neuron(weights, potential, _distanceMeasure, i);
            }
        }

        public void Train(List<double[]> input)
        {

            var edge = 0;
            for (var v = _trainVelocity; v > 0; v = v - 0.01)
            {
                for (var i = 0; i < input.Count(); i++)
                {
                    var numberTrain = edge * (input[i].Length - 1) + i + 1;

                    var winner = GetWinnerInRace(input[i]);

                    winner.CorrectWinner(input[i], v);
                    CorrectPotentialAndNeighbors(winner, input[i], v, edge);

                }

                edge++;
            }
        }



        public int ToLabel(double[] input)
        {
            return GetWinnerInRace(input, false).ClusterId;
        }

        public List<int> ToLabel(List<double[]> input, int start = 0, int end = 0, int thread = 0)
        {
            var result = new List<int>();
            end = end == 0 ? input.Count : end;

            for (var i = start; i < end; i++)
            {
                var id = GetWinnerInRace(input[i], false).ClusterId;
                if (id != 8 && id != 1)
                {
                    var t = 5;
                }
                result.Add(id);

            }

            return result;
        }

        public void ToLabel(List<double[]> input, List<int> result, int start = 0, int end = 0)
        {

            end = end == 0 ? input.Count : end;

            for (var i = start; i < end; i++)
            {
                result.Add(GetWinnerInRace(input[i], false).ClusterId);

            }


        }

        public List<List<int>> ToLabelParallel(List<double[]> input, int countThread = 8)
        {
            int countOnTask = input.Count / countThread;
            Task[] tasks = new Task[countThread];
            var result = new List<List<int>>();

            var currentWeights = Clusters.Select(x => x.Weights).ToList();

            for (var i = 0; i < countThread; i++)
            {
                var start = countOnTask * i;
                var end = i == (countThread - 1) ? input.Count : countOnTask * (i + 1);


                var copyNet = new KohonenNetwork(AmountClusters, Dimensions, _distanceMeasure.Copy(), Clusters.Select(x => x.Weights.Select(y => y).ToArray()).ToList());

                tasks[i] = new Task(() => result.Add(copyNet.ToLabel(input, start, end, i)));
                tasks[i].Start();
            }

            Task.WaitAll(tasks);

            return result;
        }

        private Neuron GetWinnerInRace(double[] vector, bool usePotential = true)
        {
            Neuron winner = Clusters[0];
            double minDistance = Clusters[0].GetDistanceToVector(vector);
            foreach (var cluster in Clusters)
            {
                if (cluster.Potential < _potentialMin && usePotential) continue;

                var distance = cluster.GetDistanceToVector(vector);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    winner = cluster;
                }
            }

            return winner;
        }

        public void CorrectPotentialAndNeighbors(Neuron winner, double[] vector, double velocity, int edge)
        {
            foreach (var cluster in Clusters)
            {
                if (cluster != winner)
                {
                    cluster.CorrectNeighbor(winner.Weights, vector, velocity, edge);
                }

                cluster.ChangePotential(cluster == winner, _potentialMin, 1.0 / Clusters.Length);
            }
        }

    }




    public class Neuron
    {
        public double[] Weights { get; private set; }
        public double Potential { get; private set; }
        public int CountWinner { get; set; }
        private IDistanceMeasure _distance;
        public int ClusterId { get; private set; }

        public Neuron(double[] weights, double potential, IDistanceMeasure distance, int clusterid)
        {
            Weights = weights;
            Potential = potential;
            CountWinner = 0;
            _distance = distance;
            ClusterId = clusterid;
        }

        public double GetDistanceToVector(double[] vector)
        {
            return _distance.Distance(Weights, vector);
        }

        public void ChangePotential(bool isWinner, double potentialMin, double diffPotential)
        {
            if (isWinner)
            {
                CountWinner++;
                Potential -= potentialMin;
                return;
            }
            Potential += diffPotential;       /////////change  ddd
        }

        public void CorrectWinner(double[] vector, double velocityTrain)
        {
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] += velocityTrain * (vector[i] - Weights[i]);
            }
        }

        public void CorrectNeighbor(double[] winnerWeights, double[] vector, double velocityTrain, double edge)
        {
            var sigma = Math.Pow(0.1 * Math.Exp(-1 * edge / 1000), 2);
            var functionOfNeighbors = Math.Exp(-1 * GetDistanceToVector(winnerWeights) / (2 * sigma));
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] += velocityTrain * functionOfNeighbors * (vector[i] - Weights[i]);
            }
        }

    }

    public interface IDistanceMeasure
    {
        double Distance(double[] vector1, double[] vector2);
        IDistanceMeasure Copy();
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

        public IDistanceMeasure Copy()
        {
            return new EuclideanDistance();
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

        public IDistanceMeasure Copy()
        {
            return new ManhattanDistance();
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

        public IDistanceMeasure Copy()
        {
            return new PowerDistance(_r, _p);
        }
    }
}
