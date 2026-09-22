"""마크다운 평가 보고서를 배포용 PDF로 변환한다.

외부 명령이나 브라우저 없이 ReportLab만 사용한다. 제목·목록·표·코드 블록을
지원하고, 한국어 글꼴을 찾지 못하면 잘못된 PDF를 만들지 않고 이유를 알린다.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("C:/Windows/Fonts/malgun.ttf"),
)
SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)] )(.*)$")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class PDFRenderError(RuntimeError):
    """글꼴·입력·렌더링 조건 때문에 PDF를 만들 수 없을 때 발생한다."""


def register_korean_font() -> str:
    """현재 OS에서 한국어를 표시할 수 있는 글꼴을 찾아 등록한다."""
    for font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ReportKorean", str(font_path), subfontIndex=0))
            return "ReportKorean"
        except Exception:
            continue
    raise PDFRenderError(
        "한국어 TrueType 글꼴을 찾지 못했다. Noto Sans CJK 또는 NanumGothic을 설치한 뒤 다시 실행하라."
    )


def inline(text: str) -> str:
    """마크다운 인라인 표기를 ReportLab Paragraph에서 안전한 텍스트로 바꾼다."""
    text = LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return html.escape(text, quote=False).replace("\n", "<br/>")


def split_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "ReportBody",
        parent=base["BodyText"],
        fontName=font,
        fontSize=9.2,
        leading=14.2,
        textColor=colors.HexColor("#202124"),
        spaceAfter=4 * mm,
        wordWrap="CJK",
        splitLongWords=True,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "ReportTitle",
            parent=body,
            fontSize=23,
            leading=31,
            textColor=colors.HexColor("#16324F"),
            alignment=TA_CENTER,
            spaceBefore=18 * mm,
            spaceAfter=16 * mm,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=body,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#16324F"),
            spaceBefore=10 * mm,
            spaceAfter=5 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=body,
            fontSize=12.5,
            leading=18,
            textColor=colors.HexColor("#24527A"),
            spaceBefore=7 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ReportH3",
            parent=body,
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#24527A"),
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle("ReportBullet", parent=body, leftIndent=2 * mm, spaceAfter=1.5 * mm),
        "code": ParagraphStyle(
            "ReportCode",
            parent=body,
            fontSize=7.5,
            leading=11,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            backColor=colors.HexColor("#F3F5F7"),
            borderPadding=4,
        ),
        "table": ParagraphStyle("ReportTable", parent=body, fontSize=7.2, leading=10, spaceAfter=0),
    }


def markdown_story(
    markdown: str,
    styles: dict[str, ParagraphStyle],
    page_width: float,
    document_title: str,
) -> list:
    """보고서에서 사용하는 마크다운 요소를 Platypus 요소로 변환한다."""
    lines = markdown.replace("\r\n", "\n").splitlines()
    story: list = [Paragraph(inline(document_title), styles["title"])]
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] = []
    in_code = False
    first_heading = True

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(x.strip() for x in paragraph)), styles["body"]))
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = [ListItem(Paragraph(inline(item), styles["bullet"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=6 * mm))
            story.append(Spacer(1, 2 * mm))
            bullets.clear()

    def flush_code() -> None:
        if code:
            story.append(Paragraph(inline("\n".join(code)), styles["code"]))
            code.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush_paragraph()
            flush_bullets()
            if in_code:
                flush_code()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code.append(line or " ")
            i += 1
            continue

        heading = HEADING.match(line)
        if heading:
            flush_paragraph()
            flush_bullets()
            level, text = len(heading.group(1)), heading.group(2)
            if first_heading and text.strip() == document_title.strip():
                first_heading = False
                i += 1
                continue
            first_heading = False
            style = styles["h1"] if level == 1 else styles["h2"] if level == 2 else styles["h3"]
            story.append(KeepTogether([Paragraph(inline(text), style), Spacer(1, 1 * mm)]))
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and SEPARATOR.match(lines[i + 1]):
            flush_paragraph()
            flush_bullets()
            rows = [split_cells(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(split_cells(lines[i]))
                i += 1
            columns = max(len(row) for row in rows)
            rows = [row + [""] * (columns - len(row)) for row in rows]
            data = [[Paragraph(inline(cell), styles["table"]) for cell in row] for row in rows]
            table = Table(data, colWidths=[page_width / columns] * columns, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), styles["table"].fontName),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 4 * mm)])
            continue

        item = LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            bullets.append(item.group(1))
            i += 1
            continue

        if not line.strip():
            flush_paragraph()
            flush_bullets()
        elif line.strip() == "---":
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 3 * mm))
        else:
            flush_bullets()
            paragraph.append(line)
        i += 1

    flush_paragraph()
    flush_bullets()
    flush_code()
    return story


def render_pdf(markdown: str, output: Path, *, title: str = "KV Cache 기술 다관점 평가 보고서") -> Path:
    """마크다운 문자열을 A4 PDF로 저장하고 경로를 반환한다."""
    if not markdown.strip():
        raise PDFRenderError("빈 보고서는 PDF로 변환할 수 없다.")
    font = register_korean_font()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    left = right = 18 * mm
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="김성현 · 양지윤 · 이지은",
    )
    styles = make_styles(font)
    story = markdown_story(markdown, styles, width - left - right, title)

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, stroke=0, fill=1)
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(left, 9 * mm, title)
        canvas.drawRightString(width - right, 9 * mm, str(doc.page))
        canvas.restoreState()

    try:
        document.build(story, onFirstPage=footer, onLaterPages=footer)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise PDFRenderError(f"PDF 렌더링 실패: {exc}") from exc
    if not output.exists() or output.stat().st_size == 0:
        raise PDFRenderError("PDF 파일이 생성되지 않았다.")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="마크다운 평가 보고서를 PDF로 변환")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    source = args.markdown.resolve()
    if not source.exists():
        parser.error(f"파일이 없다: {source}")
    output = args.output.resolve() if args.output else source.with_suffix(".pdf")
    print(render_pdf(source.read_text(encoding="utf-8"), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
