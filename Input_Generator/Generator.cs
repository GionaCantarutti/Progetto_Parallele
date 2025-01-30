using System.Drawing;

namespace Generator {
    public class Gen {

        public static String[] GetLines() {
            String[] list = { "First line", "Second line", "Third line" };
            return list;
        }

        public static List<(bool[,], string)> GetInputs() {
            List<(bool[,], string)> inputs = new List<(bool[,], string)>();

            inputs.Add((AllBlack(1000, 1000), "all_black"));
            inputs.Add((RandomNoise(1000,1000), "random_noise"));
            inputs.Add((RandomStreaks(1000,1000,1f/20f, 1f/20f), "random_streaks"));
            inputs.Add((RandomStreaks(1000,1000,1f/20f, 1f/60f), "sparse_streaks"));
            inputs.Add((ChessBoard(1000, 1000), "chessboard"));
            inputs.Add((Snake(1000, 1000), "long_snake"));

            return inputs;
        }

        private static Color[,] Colorify(bool[,] bw) {

            Color[,] colored = new Color[bw.GetLength(0), bw.GetLength(1)];

            for (int i = 0; i < bw.GetLength(0); i++) {
                 for (int j = 0; j < bw.GetLength(1); j++) {
                    colored[i,j] = bw[i,j] ? Color.White : Color.Black;
                 }
            }

            return colored;

        }

        public static bool[,] AllBlack(int width, int height) {

            return new bool[width,height];

        }

        public static bool[,] RandomNoise(int width, int height) {

            bool[,] input = AllBlack(width, height);

            Random rnd = new Random();

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    input[i,j] = rnd.Next(0,2) == 0;
                }
            }

            return input;

        }

        public static bool[,] RandomStreaks(int width, int height, float switchChanceA, float switchChanceB) {

            bool[,] input = AllBlack(width, height);

            Random rnd = new Random();

            bool state = false;

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    double r = rnd.NextDouble();
                    float c = state ? switchChanceA : switchChanceB;
                    if (r < c) state = !state;
                    input[i,j] = state;
                }
            }

            return input;

        }

        public static bool[,] ChessBoard(int width, int height) {
            bool[,] chess = new bool[width,height];

            for (int i = 0; i < width; i++) {
                bool state = i % 2 == 0;
                for (int j = 0; j < height; j++) {
                    chess[i,j] = state;
                    state = !state;
                }
            }

            return chess;

        }

        public static bool[,] Snake(int width, int height) {
            bool[,] chess = AllBlack(width, height);

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    if (j % 2 == 1) {
                        chess[i,j] = true;
                    } else {
                        if ((i == 0 && (j/2) % 2 == 0)|| i == height - 1 && (j/2) % 2 == 1) chess[i,j] = true;
                    }
                }
            }

            return chess;
        }

    }
}