변수: {technology} {tech_name} {criterion_id} {criterion_name} {criterion_question} {direction} {tech_summary} {source_type}

## 검색 질의 생성

조사 대상 기술: {technology} ({tech_name})
평가 기준: {criterion_id} {criterion_name}
평가 질문: {criterion_question}
검색 방향: {direction}
검색 도구: {source_type} (paper = 해당 기술 논문 전문 검색, web = 웹 검색)

기술 요약:
{tech_summary}

위 기준에 답할 근거를 찾기 위한 검색 질의를 두 개 만든다.
- effect: 효과·채택을 뒷받침하는 근거를 찾는 질의
- limitation: 한계·반대 근거·비용을 찾는 질의

규칙:
- 검색 도구가 paper이면 영어로, 논문 본문에 나올 법한 용어(예: KV cache, latent, compression, CXL, tiered memory)를 쓴다.
- web이면 영어로 쓰되 기술을 특정하는 이름(예: "DeepSeek-V2 MLA", "SK hynix ITME CXL")을 반드시 포함한다.
- 질의는 한 줄, 15단어 이내. 질문문이 아니라 검색어 나열도 좋다.
- 다른 기술과 비교하는 표현은 넣지 않는다.
