using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace ArcFaceBackend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class FaceController : ControllerBase
    {
        private readonly InferenceSession _session;

        public FaceController()
        {
            // Cargar el modelo ArcFace
            _session = new InferenceSession("obj/Models/arcface.onnx");
        }

        [HttpPost("compare")]
        public IActionResult CompareFaces([FromForm] IFormFile image1, [FromForm] IFormFile image2)
        {
            if (image1 == null || image2 == null)
            {
                return BadRequest("Ambas imágenes son necesarias.");
            }

            try
            {
                // Convertir las imágenes a embeddings
                var embedding1 = GetFaceEmbedding(image1);
                var embedding2 = GetFaceEmbedding(image2);

                if (embedding1 == null || embedding2 == null)
                {
                    return BadRequest("No se detectaron rostros en una o ambas imágenes.");
                }

                // Calcular la similitud coseno entre los embeddings
                var similarity = CalculateCosineSimilarity(embedding1, embedding2);

                return Ok(new { similarity = similarity * 100 }); // Retorna el porcentaje de similitud
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Error interno: {ex.Message}");
            }
        }

        private float[] GetFaceEmbedding(IFormFile imageFile)
        {
            using var stream = imageFile.OpenReadStream();
#pragma warning disable CA1416 // Validate platform compatibility
            using var bitmap = new Bitmap(stream);
#pragma warning restore CA1416 // Validate platform compatibility

            // Preprocesar la imagen (redimensionar a 112x112 y normalizar)
#pragma warning disable CA1416 // Validate platform compatibility
            var resized = new Bitmap(bitmap, new Size(112, 112));
#pragma warning restore CA1416 // Validate platform compatibility
            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 112, 112 });

            for (int y = 0; y < 112; y++)
            {
                for (int x = 0; x < 112; x++)
                {
#pragma warning disable CA1416 // Validate platform compatibility
                    var pixel = resized.GetPixel(x, y);
#pragma warning restore CA1416 // Validate platform compatibility
                    inputTensor[0, 0, y, x] = (pixel.R - 127.5f) / 128.0f; // Canal R
                    inputTensor[0, 1, y, x] = (pixel.G - 127.5f) / 128.0f; // Canal G
                    inputTensor[0, 2, y, x] = (pixel.B - 127.5f) / 128.0f; // Canal B
                }
            }

            // Ejecutar el modelo para obtener el embedding
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", inputTensor)
            };

            using var results = _session.Run(inputs);
            var embedding = results.First().AsEnumerable<float>().ToArray();

            return embedding;
        }

        private float CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            float dotProduct = 0f;
            float normA = 0f;
            float normB = 0f;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                normA += vector1[i] * vector1[i];
                normB += vector2[i] * vector2[i];
            }

            return dotProduct / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
        }
    }
}