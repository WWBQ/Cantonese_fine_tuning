import json
import re
import uuid
from datetime import datetime


def parse_wikitext(text: str) -> str:
    """
    返回：(正文, 注释列表)
    """
    # ② 提取注释，替换为 [n] 占位符
    text = _extract_refs(text)   # ✅ 接收两个返回值


    # ① 提取 <onlyinclude>
    text = _extract_onlyinclude(text)


    # ③ 处理模板
    text = _process_templates(text)

    # ④ 移除其他标签
    text = re.sub(r'<[^>]+>', '', text)


    # ⑥ 移除分类链接
    text = re.sub(r'\[\[(分类|Category|File|档案):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # ⑦ 处理普通链接
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # ⑧ 移除粗体/斜体
    text = re.sub(r"'{2,3}", '', text)

    # ⑨ 清理空白
    text = re.sub(r'\n{3,}', '\n', text)
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)  # ✅ 去掉每行行首空格
    text = text.strip()
    return text


def _extract_refs(text: str) -> str:   # ✅ 类型注解修正
    """
    把 <ref>...</ref> 替换为 [n]
    返回：(替换后的文本, 注释列表)
    """
    footnotes = []
    counter = [0]

    def replace_ref(m):
        content = m.group(1).strip()
        content = _process_templates(content)
        content = re.sub(r'<[^>]+>', '', content)

        counter[0] += 1
        footnotes.append(f"[{counter[0]}] {content}")
        return f"[{counter[0]}]"

    # 处理 <ref>...</ref>
    text = re.sub(
        r'<ref[^>]*>(.*?)</ref>',
        replace_ref,
        text,
        flags=re.DOTALL
    )

    # 处理自闭合 <ref name="..."/>
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'\{\{reflist\}\}', '\n'.join(footnotes), text, flags=re.IGNORECASE)
    return text


def _extract_onlyinclude(text: str) -> str:
    """
    去掉 <onlyinclude> 标签，内容全部保留
    """
    text = re.sub(r'<onlyinclude>(.*?)</onlyinclude>', r'\1', text, flags=re.DOTALL)
    return text


def _process_templates(text: str) -> str:
    """
    递归处理模板，从最内层开始
    """
    # 反复处理直到没有模板为止
    max_iter = 20
    for _ in range(max_iter):
        new_text = _process_one_pass(text)
        if new_text == text:
            break
        text = new_text
    return text


def _process_one_pass(text: str) -> str:
    """
    处理最内层的模板（不含嵌套）
    """
    pattern = r'\{\{([^{}]+)\}\}'

    def replace_template(m):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split('|')]
        name = parts[0].strip().lower()
        args = parts[1:]

        return _dispatch_template(name, args)

    return re.sub(pattern, replace_template, text)


def _dispatch_template(name: str, args: list) -> str:
    """
    根据模板名决定如何处理
    """

    # ── 直接丢弃的模板 ──────────────────────────
    DISCARD = {
        'header', 'reflist', 'nonfree',
        'noteта', 'noteta',  # 繁简转换
        'gap', '页面',
    }
    # 分类模板：结尾是"作品"/"人物"等
    if name in DISCARD:
        return ''
    if re.search(r'(作品|人物|朝代|时期|世纪)$', name):
        return ''

    # ── 取第二个参数（显示名）──────────────────
    DISPLAY_SECOND = {'propernoun', 'proper noun'}
    if name in DISPLAY_SECOND:
        if len(args) >= 2:
            return args[1]  # {{ProperNoun|廬陵|歐陽脩}} → 歐陽脩
        elif len(args) == 1:
            return args[0]  # {{ProperNoun|滁}} → 滁
        return ''

    # ── 异体字：取第二个参数 ───────────────────
    if name in {'另', '异体字', '異體字'}:
        if len(args) >= 2:
            return args[1]  # {{另|讓|釀}} → 釀
        elif len(args) == 1:
            return args[0]
        return ''

    # ── 取第一个参数的模板 ─────────────────────
    FIRST_ARG = {
        'lang', 's', 'del',
        'center', 'right', 'left',
        'quote', 'poem',
        'small', 'big', 'large',
        'ruby',  # 注音：只取文字部分
    }
    if name in FIRST_ARG:
        return args[0] if args else ''

    # ── wikisource 跨wiki ──────────────────────
    if name in {'wikisource', 'ws'}:
        return ''

    # ── 兜底：有参数取最后一个，没有参数丢弃 ──
    if args:
        return args[-1]
    return ''


if __name__ == '__main__':
    date_str = datetime.now().strftime("%y%m%d")
    with open('cleaned_wiki.jsonl', 'w', encoding='utf-8') as wf:
        with open('./res.jsonl', 'r',encoding='utf-8') as rf:
            for line in rf:
                line = line.strip()
                data = json.loads(line)
                uuid_line = str(uuid.uuid4())
                meta = {
                    "uuid": uuid_line,
                    "language": data['language'],
                    "source_id": 9,
                    "url": data['url'],
                    "host": "zh.wikisource.org",
                    "extra": json.dumps({
                        "id": data['id'],
                        "title": data['title'],
                        "raw_content": data['raw_content']
                    }, ensure_ascii=False)
                }

                res = {
                    "id": uuid_line,
                    "text": data['title'] + '\n' + parse_wikitext(data['raw_content']),
                    "source": f"zh_wiki_raw_{date_str}",
                    "meta": json.dumps(meta, ensure_ascii=False)
                }
                # print(json.dumps(data, ensure_ascii=False))
                wf.write(json.dumps(res, ensure_ascii=False) + '\n')
