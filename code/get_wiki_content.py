import os
import json
import bz2
import py7zr
import io
from tqdm import tqdm
from urllib.parse import quote
import xml.etree.ElementTree as etree




def extract_content(filepath, language='zh'):
    context = etree.iterparse(decode_open(filepath), events=('end',))
    for unused_event, elem in context:
        try:
            if not elem.tag.endswith("page"):
                continue
            namespace = elem.tag[:-4]
            title = elem.find(f"./{namespace}title").text
            ns = elem.find(f"./{namespace}ns").text
            id_ = elem.find(f"./{namespace}id").text
            red_ = elem.find(f"./{namespace}redirect")

            if red_ is not None:
                elem.clear()
                continue

            if ns != "0":
                elem.clear()
                continue

            all_revision = elem.iterfind(f"./{namespace}revision")
            last_revision = None
            for revision in all_revision:
                last_revision = revision

            if not last_revision:
                elem.clear()
                continue

            raw_content = last_revision.find(f"./{namespace}text").text
            elem.clear()
            yield (id_, title, raw_content)
        except EOFError as e:
            continue


def construct_url(title, language='zh'):
    # See: https://meta.wikimedia.org/wiki/Help:URL
    return f"https://{language}.wikisource.org/wiki/{quote(title)}"


def get_results(generator, project, language):
    results = []
    # for inputs in generator:
    for inputs in tqdm(generator, desc=f"Parsing {language}:"):
        try:
            id_, title, raw_content = inputs
            if not id_ or not title or not raw_content:
                continue
            url = construct_url(title, language)
            ret = {
                "id": id_,
                "url": url,
                "title": title,
                "raw_content": raw_content,
                "language": language,
                'project': project,
            }
            yield json.dumps(ret, ensure_ascii=False)
        except EOFError as e:
            continue


def decode_open(filename, mode='rt', encoding='utf-8'):
    """
    Open a file, decode and decompress, depending on extension `gz`, `bz2`, or `7z`.
    :param filename: the file to open.
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        return gzip.open(filename, mode, encoding=encoding)
    elif ext == '.bz2':
        return bz2.open(filename, mode=mode, encoding=encoding, errors='ignore')
    elif ext == '.7z':
        archive = py7zr.SevenZipFile(filename, mode='r')
        for name, bio in archive.readall().items():
            return io.TextIOWrapper(bio, encoding=encoding, errors='ignore')
    else:
        return open(filename, mode, encoding=encoding)


if __name__ == '__main__':
    all_projects = ['wikisource', 'wiktionary', 'wikibooks', 'wikiquote',
                    'wikinews', 'wikivoyage', 'wikiversity']
    base_path = r"C:\Users\v_ppzzwang\Downloads"
    files = os.listdir(base_path)
    for input_file in files:
        project = ''
        if input_file.endswith('.xml.bz2'):
            for p in all_projects:
                if p in input_file:
                    project = p
            print(input_file)
            print(project)
            language = input_file.split('wiki')[0]
            print(language)
            generator = extract_content(os.path.join(base_path,input_file), language)
            results = get_results(generator, project, language)
            path = f"../raw_content/{language}_{project}_raw_content.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result + '\n')