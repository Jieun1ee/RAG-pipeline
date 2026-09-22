변수: {technology} {criterion} {axes} {levels} {findings}

## 수준 판정

대상 기술: {technology}

평가 기준:
{criterion}

판정 축: {axes}

수준 정의:
{levels}

이 기준에 대해 수집된 Finding (id, 방향, 주장, 근거 요약):
{findings}

Finding만 근거로 판정한다.
- 판정 축마다 수준 정의 중 하나를 골라 levels에 {{axis, level}}로 넣는다. level은 수준 정의의 키를 그대로 쓴다.
- 판정 축이 없으면 levels는 빈 목록으로 두고 rationale만 쓴다.
- Finding이 판정하기에 부족하면 status=insufficient_evidence, 기준이 이 기술에 해당하지 않으면 not_applicable. 그 외 assessed.
- rationale은 한국어 2~4문장. 어느 Finding이 근거인지 id로 언급한다. 다른 기술과 비교하지 않는다.
