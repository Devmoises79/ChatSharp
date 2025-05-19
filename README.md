### 🤖 Chat de IA com ML.NET
Fala Dev!

Este é um projeto simples de chat interativo com inteligência artificial, criado em **C#** utilizando a biblioteca **ML.NET**. O sistema identifica a intenção do usuário com base na frase digitada e responde automaticamente de acordo com a previsão feita pelo modelo treinado.
Além disso, essa é uma das maneiras de praticar e aprimorar o conhecimento em programação em uma linguagem versátil e potente como é o C# (através da platamorfa .Net).

---

## 📚 Funcionalidades

- Classificação de intenções com base no texto do usuário.
- Treinamento do modelo diretamente no código com um conjunto de dados embutido.
- Respostas automáticas conforme a intenção detectada.
- Executado diretamente no **terminal/console**.

---

## 🔍 Intenções disponíveis

O modelo consegue identificar as seguintes intenções:

- `Saudacao`
- `Ajuda`
- `Curiosidade`
- `Agradecimento`
- `Frustracao`
- `Comemoracao`

---

## 🚀 Como executar

- 1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/chat-ia-mlnet.git
cd chat-ia-mlnet
```

- 2. **Compile o projeto no terminal:**

```bash
dotnet build
```

- 3. **Execute o projeto:**

```bash
dotnet run
```

## 🧠 Como funciona
 - O modelo de machine learning é treinado usando o algoritmo SdcaMaximumEntropy do ML.NET.

- As frases e intenções estão embutidas diretamente no código-fonte como uma lista.

- Após o treinamento, o sistema usa o modelo para prever a intenção de novas frases digitadas no chat.

## 🛠 Tecnologias utilizadas

- .NET 6 ou superior

- ML.NET (ferramenta da Microsoft que permite que você crie e use inteligência artificial dentro de programas em C# ou .NET, sem precisar sair do ambiente que já está acostumado.)

## ✨ Exemplo de uso

```csharp

Você: Oi  
[IA] Intenção: Saudacao  
[IA] Olá! Como posso te ajudar?

Você: Me ajuda a aprender a linguagem C# (csharp)
[IA] Intenção: Ajuda  
[IA] Claro! Estou aqui para te ajudar com programação.
```

# Autor: [@Devmoises79] 👨🏾‍💻☕


