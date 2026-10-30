"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --imode    # 会話履歴を保って対話する

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)

    return response.text


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is None:
        print("[トークン数] 使用量情報がありません")
    else:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(
            f"[入力トークン数] "
            f"{input_tokens if input_tokens is not None else '不明'}"
        )
        print(
            f"[出力トークン数] "
            f"{output_tokens if output_tokens is not None else '不明'}"
        )


def ask_interactive() -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    print("対話モードです。終了するには /exit または Ctrl+D を入力してください。")
    while True:
        try:
            question = input("\nあなた> ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if not question:
            continue

        try:
            response = chat.send_message(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue

        print_usage(response)
        print(response.text or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--imode", action="store_true", help="会話履歴を保って対話する")
    args = parser.parse_args()

    if args.imode and args.question:
        parser.error("--imode と質問の直接指定は併用できません")

    print(f"[model] {MODEL}")
    try:
        if args.imode:
            ask_interactive()
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
