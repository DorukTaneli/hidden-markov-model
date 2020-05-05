using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace HMM
{
    class Program
    {
        public const int BW_ITER = 10000;

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
                        Console.WriteLine("\nLog probability: " + Math.Log(p1));
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
                        Console.WriteLine("Log probability: " + Math.Log(prob));

                        int transitions = 0;
                        for (int i = 1; i < stateSequence.Length; i++) if (stateSequence[i] != stateSequence[i - 1]) transitions++;
                        Console.WriteLine("Number of state transitions: " + transitions);
                        break;

                    case "learn":
                        obsList = LoadObservationList();
                        WriteObservationList(obsList);
                        Console.WriteLine("Learning with ^this^ data. Please wait.");

                        model = Learn(obsList);
                        WriteModeltoFile(model);
                        WriteModel(model);
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

            Console.WriteLine("\nPi: ");
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

                sw.WriteLine("\nPi: ");
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
            int N = A.GetLength(0); //length of state-graph
            int T = obs.Length; //length of observations
            double[,] forward = new double[N, T]; //probability matrix

            //initialization step
            for (int s = 0; s < N; s++) //for each state s
                forward[s, 0] = pi[s] * B[s, obs[0]];

            //recursion step
            for (int t = 1; t < T; t++) //for each time step t
            {
                for (int s = 0; s < N; s++) //for eash state s
                {
                    double sum = 0.0;
                    for (int si = 0; si < N; si++) //si: s'
                        sum += forward[si, t - 1] * A[si, s];
                    forward[s, t] = sum * B[s, obs[t]];
                }
            }

            //termination step
            double forwardprob = 0.0;
            for (int s = 0; s < N; s++) 
                forwardprob += forward[s, T - 1];

            return forwardprob;
        }

        //Algorithm for Part2
        static int[] Viterbi(HiddenMarkovModel model, int[] obs, out double bestpathprob)
        {
            double[,] A = model.A;
            double[,] B = model.B;
            double[] pi = model.Pi;
            int N = A.GetLength(0); //length of state-graph
            int T = obs.Length; //length of observations
            double[,] viterbi = new double[N, T]; //path probability matrix
            int[,] backpointer = new int[N, T];
            int bestpathpointer;
            int[] bestpath = new int[T];

            //initialization step
            for (int i = 0; i < N; i++) //for each state s
            {
                viterbi[i, 0] = pi[i] * B[i, obs[0]];
                backpointer[i, 0] = 0;
            }

            //recursion step
            for (int t = 1; t < T; t++) //for each time step t
            {
                for (int s = 0; s < N; s++) //for each state s
                {
                    double max = 0;
                    int argmax = 0;
                    double tmp;

                    for (int si = 0; si < N; si++) //si: s'
                    {
                        tmp = viterbi[si, t - 1] * A[si, s];
                        if (tmp > max)
                        {
                            max = tmp;
                            argmax = si;
                        }
                    }

                    viterbi[s, t] = max * B[s, obs[t]];
                    backpointer[s, t] = argmax;
                }
            }

            //termination step
            double max2 = 0;
            int argmax2 = 0;
            double tmp2;

            for (int s = 0; s < N; s++)
            {
                tmp2 = viterbi[s, T - 1];
                if (tmp2 > max2)
                {
                    max2 = tmp2;
                    argmax2 = s;
                }
            }
            bestpathprob = max2;
            bestpathpointer = argmax2;

            bestpath[T - 1] = bestpathpointer;
            for (int t = T - 2; t >= 0; t--) 
                bestpath[t] = backpointer[bestpath[t + 1], t + 1];

            return bestpath;
        }

        //Algorithms for Part3
        static HiddenMarkovModel Learn(List<int[]> obsList)
        {
            int L = obsList.Count; //observation sequence count

            //number of states = no of unique observations
            int N = obsList.SelectMany(a => a).Distinct().Count();

            // Estimate Pi from first element of observations
            double[] pi = new double[N];
            for (int i = 0; i < N; i++)
            {
                foreach (int[] obs in obsList) 
                    if (obs[0] == i) 
                        pi[i]++;
                pi[i] /= L;
            }

            Random r = new Random(); // Random for initial A and B

            //initial random values for A
            double[,] A = new double[N, N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A[i, j] = r.Next();

            //normalize A
            for (int i = 0; i < N; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < N; j++)
                    sum += A[i, j];

                for (int j = 0; j < N; j++)
                    A[i, j] /= sum;
            }

            //initial random values for B
            double[,] B = new double[N, N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    B[i, j] = r.Next();

            //normalize B
            for (int i = 0; i < N; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < N; j++) 
                    sum += B[i, j];
                for (int j = 0; j < N; j++)
                    B[i, j] /= sum;
            }

            //Loop Baum-Welch for BW_ITER times
            for (int iter = 0; iter < BW_ITER; iter++)
            {
                List<double[,]> gammaList = new List<double[,]>();
                List<double[,,]> epsilonList = new List<double[,,]>();

                foreach (int[] obs in obsList) //foreach observation sequence
                {
                    int T = obs.Length;

                    //Calculate the forward and backward probability for each HMM state.
                    double[,] fwd = BWForward(A, B, pi, obs);
                    double[,] bwd = BWBackward(A, B, pi, obs);

                    double prob = 0.0;
                    for (int i = 0; i < N; i++) 
                        prob += fwd[T - 1, i];

                    //Determine the frequency of the transition-emission pair values
                    //and divide by the probability of the entire string.
                    double[,] gamma = new double[T, N];
                    for (int i = 0; i < T; i++)
                        for (int j = 0; j < N; j++)
                            gamma[i, j] = fwd[i, j] * bwd[i, j] / prob;

                    gammaList.Add(gamma);

                    double[,,] epsilon = new double[T, N, N];
                    for (int t = 0; t < T - 1; t++)
                        for (int i = 0; i < N; i++)
                            for (int j = 0; j < N; j++)
                                epsilon[t, i, j] = fwd[t, i] * A[i, j] * B[j, obs[t + 1]] * bwd[t + 1, j] / prob;

                    epsilonList.Add(epsilon);

                }

                //Re-estimate A
                for (int i = 0; i < N; i++) //N: state count
                {
                    for (int j = 0; j < N; j++)
                    {
                        double num = 0.0, den = 0.0;
                        for (int l = 0; l < L; l++) //L: observation sequence count
                        {
                            for (int t = 0; t < obsList[0].Length; t++)
                            {
                                num += epsilonList[l][t, i, j];
                                den += gammaList[l][t, i];
                            }
                        }
                        A[i, j] = num / den;
                    }
                }
                //Re-estimate B
                for (int j = 0; j < N; j++) //N: state count
                {
                    for (int k = 0; k < N; k++)
                    {
                        double num = 0.0, den = 0.0;
                        for (int l = 0; l < L; l++) //observation sequence count
                        {
                            for (int t = 0; t < obsList[0].Length; t++)
                            {
                                if (obsList[l][t] == k) 
                                    num += gammaList[l][t, j];
                                den += gammaList[l][t, j];
                            }
                        }
                        B[j, k] = num / den;
                    }
                }
            } //end of BW Loop

            return new HiddenMarkovModel(A, B, pi);
        }

        //Forward probabilities function to be used in learning
        private static double[,] BWForward(double[,] A, double[,] B, double[] pi, int[] obs)
        {
            int N = A.GetLength(0); //length of state-graph
            int T = obs.Length; //length of observations
            double[,] fwd = new double[T, N]; //probability matrix

            //initialization step
            for (int i = 0; i < N; i++) //for each state i
                fwd[0, i] = pi[i] * B[i, obs[0]];

            //recursion step
            for (int t = 0; t < T - 1; t++) //for each time step t
            {
                for (int j = 0; j < N; j++) //for each state j
                {
                    double sum = 0;
                    for (int i = 0; i < N; i++) //i: s'
                        sum += fwd[t, i] * A[i, j];
                    fwd[t + 1, j] = sum * B[j, obs[t + 1]];
                }
            }

            return fwd;
        }

        //Backward probabilities function to be used in learning
        static double[,] BWBackward(double[,] A, double[,] B, double[] pi, int[] obs)
        {

            int N = A.GetLength(0); //length of state-graph
            int T = obs.Length; //length of observations
            double[,] bwd = new double[T, N]; //probability matrix

            //initialization step 
            for (int i = 0; i < N; i++) //for each state i
                bwd[T - 1, i] = 1;

            //recursion step
            for (int t = T - 2; t >= 0; t--) //for each time step t
            {
                for (int i = 0; i < N; i++) //for each state i
                {
                    bwd[t, i] = 0;
                    for (int j = 0; j < N; j++) 
                        bwd[t, i] += A[i, j] * B[j, obs[t + 1]] * bwd[t + 1, j];
                }
            }

            return bwd;
        }
    }

    //HiddenMarkovModel class that holds a 
    class HiddenMarkovModel
    {
        public double[,] A;
        public double[,] B;
        public double[] Pi;

        //constructor from model.txt file
        public HiddenMarkovModel(List<String> lines)
        {
            // find beginning of matrices
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

        //constructor using matrices
        public HiddenMarkovModel(double[,] A, double[,] B, double[] Pi) {
            this.A = A;
            this.B = B;
            this.Pi = Pi;
        }
    }
}
