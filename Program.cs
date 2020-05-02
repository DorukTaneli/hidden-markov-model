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
            // Init vars for the HMM and observation sequence
            HiddenMarkovModel model;
            int[] obs;

            // Init var to read user input
            string command;

            // Title and help text
            string usage = @"
            Commands
            --------
                          
            obsv_prob        Part 1: calculates probability of observation sequence given model
            viterbi          Part 2: calculates best state sequence given observation and model, 
                                     Probability of the observation given model and state sequence,
                                     and number of state transitions";

            Console.WriteLine(usage);

            while (true)
            {
                Console.Write("\n>");
                command = Console.ReadLine();

                switch (command)
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

                    default:
                        Console.WriteLine("Command list is at top");
                        break;
                }

            }

        }

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

    }


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
    }
}
