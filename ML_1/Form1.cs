using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML_1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var mlContext = new MLContext();

            if(openFileDialog1.ShowDialog()==DialogResult.OK)
            {
                var textLoader = mlContext.Data.CreateTextLoader(
                    columns: new[]
                    {
                        new TextLoader.Column("Features",DataKind.Single,0,3),
                        new TextLoader.Column("iLabel",DataKind.String,4)
                    },
                    separatorChar: ',',
                    hasHeader: true
                    );

                var dv = textLoader.Load(openFileDialog1.FileName);

                var splitData = mlContext.Data.TrainTestSplit(dv, 0.4);

                IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "iLabel")
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                ITransformer model = estimator.Fit(splitData.TrainSet);

                IDataView predictions = model.Transform(splitData.TestSet);

                var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

                writeResulttoFile(metrics);

                string modelPath = Path.Combine(Path.GetDirectoryName(openFileDialog1.FileName), "LightGBM.mdl");
                mlContext.Model.Save(model, splitData.TrainSet.Schema, modelPath);

                DataViewSchema predSchema;
                ITransformer loadedModel = mlContext.Model.Load(modelPath, out predSchema);

                var textLoader2 = mlContext.Data.CreateTextLoader(
                    columns: new[]
                    {
                        new TextLoader.Column("Features",DataKind.Single,0,3),
                        new TextLoader.Column("iLabel",DataKind.String,4)
                    },
                    separatorChar: ',',
                    hasHeader: false
                    );

                var predictionOut = loadedModel.Transform(textLoader2.Load(Path.Combine(Path.GetDirectoryName(openFileDialog1.FileName), "test.csv")));
                var view = predictionOut.Preview(); 
                string[] labelColumn = predictionOut.GetColumn<string>("PredictedLabel").ToArray();



                MessageBox.Show("OK " + labelColumn[0]);

            }
        }

        private void writeResulttoFile(MulticlassClassificationMetrics metrics)
        {
            string result = ("Micro Accuracy: " + metrics.MicroAccuracy.ToString("##.###")
                    + "\nMacro Accuracy: " + metrics.MacroAccuracy.ToString("##.###")
                    + "\nConfMat: " + metrics.ConfusionMatrix.GetFormattedConfusionTable()
                    );
            using (var outfs = new StreamWriter(Path.Combine(Path.GetDirectoryName(openFileDialog1.FileName),"result.txt")))
            {
                outfs.Write(result);
            }
        }
    }
}
