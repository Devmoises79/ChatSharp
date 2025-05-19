using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

public class MensagemData
{
    public string Texto { get; set; }
    public string Intent { get; set; }
}


public class MensagemPredicao
{
    [ColumnName("PredictedLabel")]
    public string IntentPrevista { get; set; }
}

    public class Program
        {
        static void Main()
        {
            var contextoML = new MLContext();

            var dados = new List<MensagemData>
            {
            // Saudação
            new() { Texto = "Oi", Intent = "Saudacao" },
            new() { Texto = "Olá", Intent = "Saudacao" },
            new() { Texto = "Boa tarde", Intent = "Saudacao" },
            new() { Texto = "E aí", Intent = "Saudacao" },
            new() { Texto = "Tudo bem?", Intent = "Saudacao" },

            // Ajuda
            new() { Texto = "Preciso de ajuda com programação", Intent = "Ajuda" },
            new() { Texto = "Me ajuda com C#", Intent = "Ajuda" },
            new() { Texto = "Como uso ASP.NET?", Intent = "Ajuda" },
            new() { Texto = "Qual comando usar no terminal?", Intent = "Ajuda" },

            // Curiosidade
            new() { Texto = "Quem é você?", Intent = "Curiosidade" },
            new() { Texto = "Qual seu nome?", Intent = "Curiosidade" },
            new() { Texto = "Você é inteligente?", Intent = "Curiosidade" },
            new() { Texto = "O que você faz?", Intent = "Curiosidade" },

            // Agradecimento
            new() { Texto = "Obrigado", Intent = "Agradecimento" },
            new() { Texto = "Valeu", Intent = "Agradecimento" },
            new() { Texto = "Agradeço pela ajuda", Intent = "Agradecimento" },

            // Frustração
            new() { Texto = "Isso não funcionou", Intent = "Frustracao" },
            new() { Texto = "Está dando erro", Intent = "Frustracao" },
            new() { Texto = "Não entendi nada", Intent = "Frustracao" },
            new() { Texto = "Que confuso", Intent = "Frustracao" },

            // Comemoração
            new() { Texto = "Consegui!", Intent = "Comemoracao" },
            new() { Texto = "Deu certo!", Intent = "Comemoracao" },
            new() { Texto = "Funcionou!", Intent = "Comemoracao" },
            new() { Texto = "Obrigado!", Intent = "Comemoracao" }
            };


        var dadosTreino = contextoML.Data.LoadFromEnumerable(dados);

        var pipeline = contextoML.Transforms.Conversion.MapValueToKey("Label", nameof(MensagemData.Intent))
            .Append(contextoML.Transforms.Text.FeaturizeText("Features", nameof(MensagemData.Texto)))
            .Append(contextoML.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(contextoML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var modelo = pipeline.Fit(dadosTreino);

        var engine = contextoML.Model.CreatePredictionEngine<MensagemData, MensagemPredicao>(modelo);

        Console.WriteLine("Chat de IA com ML.NET");
        Console.WriteLine("----- Bem-vindo ao chat! -----");
        Console.WriteLine("Sou uma versão simples de uma IA treinada com ML.NET. \n");
        Console.WriteLine("Minhas intenções são: Saudação, Ajuda, Curiosidade, Agradecimento, Frustração, Comemoração.");
        Console.WriteLine("Por enquanto sou um chat simples, mas, não se preocupe pois irei melhorar futuramente.");
        Console.WriteLine("Digite uma mensagem para a IA ou 'sair' para encerrar.");
        

        while (true)
        {
            Console.Write("\n Você: ");
            string entrada = Console.ReadLine();

            if (entrada?.ToLower() == "sair")
                break;

            var predicao = engine.Predict(new MensagemData { Texto = entrada });

            Console.WriteLine($"[IA] Intenção: {predicao.IntentPrevista}");

            switch (predicao.IntentPrevista)
        {
            case "Saudacao":
                Console.WriteLine("[IA] Olá! Como posso te ajudar?");
                break;
            case "Ajuda":
                Console.WriteLine("[IA] Claro! Estou aqui para te ajudar com programação.");
                break;
            case "Curiosidade":
                Console.WriteLine("[IA] Eu sou uma IA treinada com ML.NET, me chamo Lucy. Posso responder suas dúvidas!");
                break;
            case "Agradecimento":
                Console.WriteLine("[IA] De nada! Fico feliz em ajudar 😊");
                break;
            case "Frustracao":
                Console.WriteLine("[IA] Sinto muito! Vamos tentar resolver isso juntos?");
                break;
            case "Comemoracao":
                Console.WriteLine("[IA] Uhul! Fico feliz que deu certo! 🎉");
                break;
            default:
                Console.WriteLine("[IA] Ainda não entendi bem... pode reformular?");
            break;
        }

        }

        Console.WriteLine("\n👋 Até logo!");
    }
}

                
                