"""모델 호출을 한곳으로 모은 창구.

부르는 쪽은 generate()와 judge() 둘만 쓴다. 응답을 Pydantic 모델로 받는 구조화 출력, 실패 시
한 번 재요청, 파일 캐시가 모두 여기서 처리된다. 두 함수는 config.yaml에서 읽는 모델만 다르다.
모델은 init_chat_model로 만든다. 공급자별 클래스를 직접 부르지 않아 모델을 바꿀 때 이름만 고치면 된다.
생성과 판정에 서로 다른 모델을 써야 판정이 생성을 그대로 승인하는 일을 줄일 수 있다.

    python -m core.llm --list     API 키로 접근 가능한 모델 id를 출력한다
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TypeVar

from pydantic import BaseModel

from core import cache, config

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# 추론 계열 모델은 temperature 인자를 받지 않아 넘기면 오류가 난다
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class LLMUnavailable(RuntimeError):
    """모델을 부를 수 없는 상태. 외부 호출을 끈 모드이거나 모델 이름이 비어 있을 때 발생한다."""


def _model_name(role: str) -> str:
    """역할(generator, judge)에 해당하는 모델 이름. 부를 수 없는 상태면 예외로 이유를 알린다."""
    cfg = config.get()
    if cfg.get("dry_run"):
        raise LLMUnavailable(f"dry-run 모드에서는 LLM({role})을 호출하지 않는다. --dry-run을 빼고 실행하라.")
    name = (cfg.get("models", {}).get(role) or "").strip()
    if not name:
        raise LLMUnavailable(
            f"config.yaml의 models.{role}이 비어 있다. "
            "`python -m core.llm --list`로 접근 가능한 모델을 확인해 채워라."
        )
    return name


def _call(role: str, prompt: str, schema: type[T]) -> T | None:
    """프롬프트를 보내고 schema 형태로 파싱된 응답을 돌려준다.

    역할, 모델, 스키마, 프롬프트를 합친 문자열이 캐시 키다. 같은 조합이 이미 있으면 호출하지 않는다.
    응답이 스키마에 맞지 않으면 한 번 더 요청하고, 그래도 실패하면 None을 돌려준다.
    None은 예외가 아니라 빈 결과라서 그래프가 멈추지 않는다. 부르는 쪽이 재시도 상한 규칙으로 처리한다.
    """
    model = _model_name(role)
    cache_key = f"{role}|{model}|{schema.__name__}|{prompt}"
    cached = cache.get(cache_key)
    if cached is not None:
        return schema.model_validate(cached)

    from langchain.chat_models import init_chat_model  # 무거운 import는 실제로 부를 때만

    kwargs: dict = {"timeout": 120, "max_retries": 2}
    if not model.startswith(_NO_TEMPERATURE_PREFIXES):
        kwargs["temperature"] = config.get().get("models", {}).get("temperature", 0)
    runnable = init_chat_model(model, model_provider="openai", **kwargs).with_structured_output(schema)

    result: T | None = None
    last_error: Exception | None = None
    for _ in range(2):  # 최초 1회와 재요청 1회
        try:
            out = runnable.invoke(prompt)
            result = out if isinstance(out, schema) else schema.model_validate(out)
            break
        except Exception as exc:  # noqa: BLE001 - 스키마 불일치와 네트워크 오류 모두 재요청 대상
            last_error = exc
    if result is None:
        log.warning("%s(%s) 구조화 출력 실패, 빈 결과로 처리: %s", role, schema.__name__, last_error)
        return None
    cache.set(cache_key, result.model_dump(mode="json"))
    return result


def generate(prompt: str, schema: type[T]) -> T | None:
    """생성용 모델로 schema 형태의 응답을 만든다."""
    return _call("generator", prompt, schema)


def judge(prompt: str, schema: type[T]) -> T | None:
    """판정용 모델로 schema 형태의 응답을 받는다."""
    return _call("judge", prompt, schema)


def list_models() -> list[str]:
    """환경 변수의 API 키로 접근할 수 있는 모델 id 목록."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(override=True)
    client = OpenAI()
    return sorted(m.id for m in client.models.list())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="접근 가능한 모델 id 출력")
    args = parser.parse_args(argv)
    if args.list:
        for model_id in list_models():
            print(model_id)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
