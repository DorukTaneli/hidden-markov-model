using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace HMM
{
    class Program
    {
        static void Main(string[] args)
        {
            string commandList = 
@"Commands:      
obsv_prob       Part 1: calculates probability of observation sequence(testdata.txt) given model(model.txt)
viterbi         Part 2: calculates: 
                        best state sequence given observation(testdata.txt) and model(model.txt), 
                        probability of the observation given model(model.txt) and state sequence(newly calculated),
                        and number of state transitions for the new state sequence
learn           Part 3: creates a hidden markov model given observation list using Baum-Welch learning method,
                        writes the model to HMMdescription.txt";

            Console.WriteLine(commandList);

            HiddenMarkovModel model;
            int[] obs;
            List<int[]> obsList;

            while (true)
            {
                Console.Write("\n>");
                string input = Console.ReadLine();

                switch (input)
                {
                    case "obsv_prob":
                        model = LoadModel();
                        obs = LoadObservation();
                        double p1 = Forward(model, obs);

                        WriteModel(model);
                        WriteObservation(obs);
                        Console.WriteLine("\nProbability of the observation given model: " + p1);

                        break;

                    case "viterbi":
                        model = LoadModel();
                        obs = LoadObservation();

                        WriteModel(model);
                        WriteObservation(obs);

                        double prob;
                        int[] stateSequence = Viterbi(model, obs, out prob);
                        Console.Write("\nBest state sequence given observation and model: ");
                        for (int i = 0; i < stateSequence.Length; i++) Console.Write(stateSequence[i] + " ");

                        Console.WriteLine("\nProbability of the observation given model and state sequence: " + prob);

                        int transitions = 0;
                        for (int i = 1; i < stateSequence.Length; i++) if (stateSequence[i] != stateSequence[i - 1]) transitions++;
                        Console.WriteLine("Number of state transitions: " + transitions);
                        break;

                    case "learn":
                        obsList = LoadObservationList();
                        Console.WriteLine("Learning with the data: \n");
                        WriteObservationList(obsList);

                        Console.WriteLine("Baum-Welch Iteration: ");
                        model = Learn(obsList);
                        WriteModeltoFile(model);
                        Console.WriteLine("\nDone. Model is written to HMMdescription.txt");

                        break;

                    default:
                        Console.WriteLine(commandList);
                        break;
                }

            }

        }

        //IO Functions
        private static void WriteObservation(int[] obs)
        {
            Console.Write("\nObservation: ");

            for (int i = 0; i < obs.Length; i++)
                Console.Write(obs[i] + " ");

            Console.WriteLine();
        }

        private static void WriteModel(HiddenMarkovModel model)
        {
            Console.WriteLine("\nA:");
            for (int i = 0; i < model.A.GetLength(0); i++)
            {
                for (int j = 0; j < model.A.GetLength(1); j++)
                {
                    Console.Write(model.A[i, j] + " ");
                }
                Console.WriteLine();
            }

            Console.WriteLine("\nB:");
            for (int i = 0; i < model.B.GetLength(0); i++)
            {
                for (int j = 0; j < model.B.GetLength(1); j++)
                {
                    Console.Write(model.B[i, j] + " ");
                }
                Console.WriteLine();
            }

            Console.Write("\nPi: ");
            for (int i = 0; i < model.Pi.Length; i++)
                Console.Write(model.Pi[i] + " ");
            Console.WriteLine();
        }

        private static void WriteModeltoFile(HiddenMarkovModel model)
        {
            using (StreamWriter sw = new StreamWriter(Path.Combine(Environment.CurrentDirectory, "HMMdescription.txt"))) 
            {
                sw.WriteLine("Number of States: " + model.A.GetLength(0));

                sw.WriteLine("\nA:");
                for (int i = 0; i < model.A.GetLength(0); i++)
                {
                    for (int j = 0; j < model.A.GetLength(1); j++)
                    {
                        sw.Write(model.A[i, j] + " ");
                    }
                    sw.WriteLine();
                }

                sw.WriteLine("\nB:");
                for (int i = 0; i < model.B.GetLength(0); i++)
                {
                    for (int j = 0; j < model.B.GetLength(1); j++)
                    {
                        sw.Write(model.B[i, j] + " ");
                    }
                    sw.WriteLine();
                }

                sw.Write("\nPi: ");
                for (int i = 0; i < model.Pi.Length; i++)
                    sw.Write(model.Pi[i] + " ");
                sw.WriteLine();
            }
        }

        private static HiddenMarkovModel LoadModel()
        {
            List<String> lines = File.ReadAllLines(Path.Combine(Environment.CurrentDirectory, "model.txt")).ToList();

            HiddenMarkovModel model = new HiddenMarkovModel(lines);

            return model;
        }

        private static int[] LoadObservation()
        {
            string line = File.ReadAllText(Path.Combine(Environment.CurrentDirectory, "testdata.txt"));

            int[] obs = new int[line.Split().Length];

            for (int i = 0; i < line.Split().Length; i++)
                obs[i] = int.Parse(line.Split()[i]);

            return obs;
        }

        private static List<int[]> LoadObservationList()
        {
            List<String> lines = File.ReadAllLines(Path.Combine(Environment.CurrentDirectory, "data.txt")).ToList();

            List<int[]> obsList = new List<int[]>();

            for (int i = 0; i < lines.Count; i++)
            {
                obsList.Add(Array.ConvertAll(lines[i].Trim().Split(), int.Parse));
            }

            return obsList;
        }

        private static void WriteObservationList(List<int[]> obsList)
        {
            for (int i = 0; i < obsList.Count; i++)
            {
                for (int j = 0; j < obsList[0].Length; j++)
                    Console.Write(obsList[i][j]);
                Console.WriteLine();
            }
        }

        //Algorithm for Part1
        static double Forward(HiddenMarkovModel model, int[] obs)
        {

            double[,] A = model.A;
            double[,] B = model.B;
            double[] pi = model.Pi;
            int N = A.GetLength(0);
            int T = obs.Length;
            double prob = 0;

            // 1) Initialization
            double[,] alpha = new double[T, N];
            for (int i = 0; i < N; i++) alpha[0, i] = pi[i] * B[i, obs[0]];

            // 2) Induction
            for (int t = 0; t < T - 1; t++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < N; i++) sum += alpha[t, i] * A[i, j];
                    alpha[t + 1, j] = sum * B[j, obs[t + 1]];
                }
            }

            // Scaling
            // Implementation of scaling coefficients
            // slightly different than Rabiner tutorial
            double[] c = new double[T];
            for (int t = 0; t < T; t++)
            {
                for (int i = 0; i < N; i++)
                {
                    c[t] += alpha[t, i];
                }
            }

            // 3) Termination
            for (int i = 0; i < N; i++) prob += alpha[T - 1, i];

            // Return result
            return prob;
        }

        //Algorithm for Part2
        static int[] Viterbi(HiddenMarkovModel model, int[] obs, out double prob)
        {

            double[,] A = model.A;
            double[,] B = model.B;
            double[] pi = model.Pi;
            int T = obs.Length;
            int N = A.GetLength(0);
            double[,] delta = new double[T, N];
            int[,] psi = new int[T, N];
            int[] q = new int[T];

            // 1) Initialization
            for (int i = 0; i < N; i++)
            {
                delta[0, i] = pi[i] * B[i, obs[0]];
                psi[0, i] = 0;
            }

            // 2) Recursion
            for (int t = 1; t < T; t++)
            {
                for (int j = 0; j < N; j++)
                {

                    double max = 0;
                    double alt = 0;
                    int argmax = 0;
                    for (int i = 0; i < N; i++)
                    {
                        alt = delta[t - 1, i] * A[i, j];
                        if (alt > max)
                        {
                            max = alt;
                            argmax = i;
                        }
                    }

                    delta[t, j] = max * B[j, obs[t]];
                    psi[t, j] = argmax;

                }
            }

            // 3) Termination
            double max1 = 0;
            double alt1 = 0;
            int argmax1 = 0;
            for (int i = 0; i < N; i++)
            {
                alt1 = delta[T - 1, i];
                if (alt1 > max1)
                {
                    max1 = alt1;
                    argmax1 = i;
                }
            }
            q[T - 1] = argmax1;
            prob = max1;

            // 4) Backtracking
            for (int t = T - 2; t >= 0; t--) q[t] = psi[t + 1, q[t + 1]];

            // Return result
            return q;

        }

        //Algorithms for Part3
        static HiddenMarkovModel Learn(List<int[]> obss)
        {

            // Store number of observation sequences
            int L = obss.Count;

            // Estimate N assuming no of states = no of unique observations
            int N = obss.SelectMany(a => a).Distinct().Count();

            // Estimate Pi
            double[] pi = new double[N];
            for (int i = 0; i < N; i++)
            {
                foreach (int[] obs in obss) if (obs[0] == i) pi[i]++;
                pi[i] /= L;
            }

            // Init Random for initial estimates of A and B
            Random r = new Random();
            // Initial estimate for A
            double[,] A = new double[N, N];
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    A[i, j] = r.Next();
                }
            }
            for (int i = 0; i < N; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < N; j++) sum += A[i, j];
                for (int j = 0; j < N; j++) A[i, j] /= sum;
            }
            // Initial estimate for B
            double[,] B = new double[N, N];
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    B[i, j] = r.Next();
                }
            }
            for (int i = 0; i < N; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < N; j++) sum += B[i, j];
                for (int j = 0; j < N; j++) B[i, j] /= sum;
            }

            // Loop Baum-Welch for 10,000 times
            for (int iter = 0; iter < 10000; iter++)
            {

                Console.CursorLeft = 0;
                Console.Write((iter + 1) + " / " + 10000);

                // Init Gamma and Xi vars
                List<double[,]> gammas = new List<double[,]>();
                List<double[,,]> xis = new List<double[,,]>();

                // Produce a Gamma and Xi matrix for every observation sequence
                foreach (int[] obs in obss)
                {

                    int T = obs.Length;

                    // Calculate forward and backward vars and probability of the sequence
                    double[,] alpha = BWForward(A, B, pi, obs);
                    double[,] beta = BWBackward(A, B, pi, obs);
                    double prob = 0.0;
                    for (int i = 0; i < N; i++) prob += alpha[T - 1, i];

                    // Gamma variable
                    double[,] gamma = new double[T, N];
                    for (int t = 0; t < T; t++)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            gamma[t, i] = alpha[t, i] * beta[t, i] / prob;
                        }
                    }
                    gammas.Add(gamma);

                    // Xi variable
                    double[,,] xi = new double[T, N, N];
                    for (int t = 0; t < T - 1; t++)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            for (int j = 0; j < N; j++)
                            {
                                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, obs[t + 1]] * beta[t + 1, j] / prob;
                            }
                        }
                    }
                    xis.Add(xi);

                }

                // Adjust A
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        double num = 0.0;
                        double den = 0.0;
                        for (int l = 0; l < L; l++)
                        {
                            for (int t = 0; t < 9; t++)
                            {
                                num += xis[l][t, i, j];
                                den += gammas[l][t, i];
                            }
                        }
                        A[i, j] = num / den;
                    }
                }
                // Adjust B
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        double num = 0.0;
                        double den = 0.0;
                        for (int l = 0; l < L; l++)
                        {
                            for (int t = 0; t < obss[0].Length; t++)
                            {
                                if (obss[l][t] == k) num += gammas[l][t, j];
                                den += gammas[l][t, j];
                            }
                        }
                        B[j, k] = num / den;
                    }
                }

            }

            return new HiddenMarkovModel(A, B, pi);
      }

        //Forward probabilities function to be used in learning
        private static double[,] BWForward(double[,] A, double[,] B, double[] pi, int[] obs)
        {

            int N = A.GetLength(0);
            int T = obs.Length;

            // 1) Initialization
            double[,] alpha = new double[T, N];
            for (int i = 0; i < N; i++) alpha[0, i] = pi[i] * B[i, obs[0]];
            // 2) Induction
            for (int t = 0; t < T - 1; t++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < N; i++) sum += alpha[t, i] * A[i, j];
                    alpha[t + 1, j] = sum * B[j, obs[t + 1]];
                }
            }

            return alpha;
        }

        //Backward probabilities function to be used in learning
        static double[,] BWBackward(double[,] A, double[,] B, double[] pi, int[] obs)
        {

            int N = A.GetLength(0);
            int T = obs.Length;

            // 1) Initialization
            double[,] beta = new double[T, N];
            for (int i = 0; i < N; i++) beta[T - 1, i] = 1;
            // 2) Induction
            for (int t = T - 2; t >= 0; t--)
            {
                for (int i = 0; i < N; i++)
                {
                    beta[t, i] = 0;
                    for (int j = 0; j < N; j++) beta[t, i] += A[i, j] * B[j, obs[t + 1]] * beta[t + 1, j];
                }
            }

            return beta;
        }
    }

    //HiddenMarkovModel class that holds a 
    class HiddenMarkovModel
    {
        public double[,] A;
        public double[,] B;
        public double[] Pi;

        public HiddenMarkovModel(List<String> lines)
        {
            // Find out where data begins
            int A_index = lines.IndexOf("A");
            int B_index = lines.IndexOf("B");
            int pi_index = lines.IndexOf("Pi");

            // Get lines with data
            List<String> A_list = lines.GetRange(A_index + 2, B_index - A_index - 3);
            List<String> B_list = lines.GetRange(B_index + 2, pi_index - B_index - 3);
            String pi_string = lines[pi_index + 2];

            // Parse A
            A = new double[A_list.Count, A_list[0].Split().Length];
            for (int i = 0; i < A_list.Count; i++)
            {
                for (int j = 0; j < A_list[0].Split().Length; j++)
                {
                    A[i, j] = double.Parse(A_list[i].Split()[j]);
                }
            }

            // Parse B
            B = new double[B_list.Count, B_list[0].Split().Length];
            for (int i = 0; i < B_list.Count; i++)
            {
                for (int j = 0; j < B_list[0].Split().Length; j++)
                {
                    B[i, j] = double.Parse(B_list[i].Split()[j]);
                }
            }

            // Parse Pi
            Pi = new double[pi_string.Split().Length];
            for (int i = 0; i < pi_string.Split().Length; i++)
            {
                Pi[i] = double.Parse(pi_string.Split()[i]);
            }

        }

        public HiddenMarkovModel(double[,] A, double[,] B, double[] Pi) {
            this.A = A;
            this.B = B;
            this.Pi = Pi;
        }
    }
}
