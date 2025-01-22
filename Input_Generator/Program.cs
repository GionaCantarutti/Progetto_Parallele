using System;
using System.IO;
using Generator;

class Program {
    static void Main(string[] args) {

        //Generate images
        List<(bool[,], string)> images = Generator.Gen.GetInputs();

        foreach (var (image, name) in images) {

            // Create a string array with the lines of text
            string[] lines = toLines(image);

            // Set a variable to the Documents path.
            string path = "Inputs";

            // Write the string array to a new file named "WriteLines.txt".
            using (StreamWriter outputFile = new StreamWriter(Path.Combine(path, name + ".txt")))
            {
                foreach (string line in lines)
                    outputFile.WriteLine(line);
            }

        }
    }

    private static string[] toLines(bool[,] matrix) {

        string[] lines = new string[matrix.GetLength(1)];

        for (int y = 0; y < matrix.GetLength(1); y++) {

            string line = "";

            for (int x = 0; x < matrix.GetLength(0); x++) {

                line += matrix[x,y] ? 1 : 0;

            }

            lines[y] = line;
        }

        return lines;

    }
}