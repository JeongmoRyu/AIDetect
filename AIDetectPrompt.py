import os
import openai


def detect_ai_generated(text: str, model: str = "gpt-4"):
    """
    Uses a "killer prompt" to classify whether the given text is AI-generated or human-written.
    Returns a dict with 'label' and 'justification'.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set.")
    # api_key = ''
    openai.api_key = api_key

    # Construct prompt
    prompt = (
        "You are an expert AI-detection specialist. Analyze the following text and decide if it was written by a human or by an AI.\n"
        "- If AI, respond with exactly: `AI-generated`\n"
        "- If human, respond with exactly: `Human-written`\n"
        "Also, include a brief justification sentence after your label.\n\n"
        "Text to analyze:\n"
        f"{text}\n\n"
        "Classification:"
    )

    # Query OpenAI ChatCompletion
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=60
    )

    # Extract and return the classification with justification
    output = response.choices[0].message.content.strip()
    label, _, justification = output.partition("\n")
    return {"label": label, "justification": justification}


def main():
    # give content with parse
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect if text is AI-generated or human-written."
    )
    parser.add_argument(
        "--file", type=str, help="Path to a text file to analyze."
    )
    parser.add_argument(
        "--text", type=str, help="Text string to analyze."
    )
    args = parser.parse_args()
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
    elif args.text:
        content = args.text
    else:
        parser.error("Provide either --file or --text to analyze.")



    # give content directly
    #     content = """
    #     위 코드 예시는 OpenAI API를 활용해 “killer prompt”를 포함한 간단한 분류 함수를 구현한 것입니다.

    # detect_ai_generated(text) 함수는 입력된 텍스트를 모델에 전달하여 AI-generated 또는 Human-written 레이블과 함께 간략한 근거를 반환합니다.

    # --file 또는 --text 인자를 통해 분석할 텍스트를 지정할 수 있으며, 결과를 콘솔에 출력하도록 되어 있습니다.
    #     """
    
    result = detect_ai_generated(content)
    print(f"Label: {result['label']}")
    print(f"Justification: {result['justification']}")


if __name__ == "__main__":
    main()
