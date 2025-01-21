using System;
using System.IO;
using Generator;

class Program {
    static void Main(string[] args) {

        // Create a string array with the lines of text
        string[] lines = Generator.Gen.GetLines();

        // Set a variable to the Documents path.
        string path = "Inputs";

        // Write the string array to a new file named "WriteLines.txt".
        using (StreamWriter outputFile = new StreamWriter(Path.Combine(path, "WriteLines.txt")))
        {
            foreach (string line in lines)
                outputFile.WriteLine(line);
        }

    }
}