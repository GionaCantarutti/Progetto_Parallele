using System.Drawing;

namespace Generator {
    public class Gen {

        public static String[] GetLines() {
            String[] list = { "First line", "Second line", "Third line" };
            return list;
        }

        public static List<Color[,]> GetInputs() {
            List<Color[,]> inputs = new List<Color[,]>();

            inputs.Add(Colorify(AllBlack(1000, 1000)));
            inputs.Add(Colorify(RandomNoise(1000,1000)));
            inputs.Add(Colorify(RandomStreaks(1000,1000,1/20)));

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

        public static bool[,] RandomStreaks(int width, int height, float switchChance) {

            bool[,] input = AllBlack(width, height);

            Random rnd = new Random();

            bool state = false;

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    if (rnd.NextDouble() < switchChance) state = !state;
                    input[i,j] = state;
                }
            }

            return input;

        }

    }
}