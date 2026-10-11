"""核对报告表格、引用、来源指纹与实际分页，并生成逐页PNG用于视觉检查。"""
from pathlib import Path
import json, re, hashlib, argparse, os, shutil
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
from pdf2image import convert_from_path

ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'outputs/report_build';REPORT=ROOT/'report'
DOCX=REPORT/'CT图像重建与稀疏角度优化实验报告修订版.docx'
MD=REPORT/'CT图像重建与稀疏角度优化实验报告修订版.md'
POPPLER=Path(r'C:\Users\wsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poppler',default=os.environ.get('POPPLER_PATH'),help='Poppler的bin目录；也可设置POPPLER_PATH或加入PATH')
    args=parser.parse_args()
    poppler=args.poppler or (None if shutil.which('pdftoppm') else str(POPPLER) if POPPLER.is_dir() else None)
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with ZipFile(DOCX) as z:
        xml=etree.fromstring(z.read('word/document.xml'))
        styles=etree.fromstring(z.read('word/styles.xml'))
    tables=json.loads((BUILD/'report_tables.json').read_text(encoding='utf-8'))
    actual=xml.findall('.//w:body/w:tbl',ns);assert len(actual)==len(tables)==14
    checks=1
    for table,expected in zip(actual,tables):
        rows=[]
        for row in table.findall('w:tr',ns):
            rows.append([''.join(c.xpath('.//w:t/text()',namespaces=ns)) for c in row.findall('w:tc',ns)])
        assert rows==[[str(v) for v in r] for r in [expected['headers']]+expected['rows']],expected['caption'];checks+=1
    text=MD.read_text(encoding='utf-8')
    refs=json.loads((BUILD/'report_bibliography.json').read_text(encoding='utf-8'))
    used=set(int(x) for x in re.findall(r'\[(\d+)\]',text.split('## 参考文献')[0]))
    assert used==set(range(1,17));checks+=1
    for r in refs:assert f"[{r['number']}] {r['text']}" in text;checks+=1
    figures=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    for p in figures:assert (REPORT/p).is_file();checks+=1
    assert len(xml.findall('.//m:oMath',ns))==7;checks+=1
    assert '_' not in ''.join(xml.xpath('.//m:t/text()',namespaces=ns));checks+=1
    assert not styles.findall('.//w:pBdr',ns);checks+=1
    assert '[[TOC]]' not in ''.join(xml.xpath('.//w:t/text()',namespaces=ns));checks+=1
    assert xml.xpath('.//w:instrText[contains(text(),"TOC")]',namespaces=ns);checks+=1
    for r in xml.xpath('.//w:r[w:t]',namespaces=ns):
        rf=r.find('w:rPr/w:rFonts',ns)
        if rf is not None:
            for key,expected in [('eastAsia','宋体'),('ascii','Times New Roman'),('hAnsi','Times New Roman')]:
                actual_font=rf.get('{'+ns['w']+'}'+key)
                assert actual_font is None or actual_font==expected,(key,actual_font)
    checks+=1
    for row in json.loads((ROOT/'outputs/final/sources.json').read_text(encoding='utf-8'))['sources']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path'];checks+=1
    pdf=BUILD/'render/report.pdf';reader=PdfReader(pdf);pages=[]
    for i,page in enumerate(reader.pages,1):
        t=page.extract_text();assert len(t.strip())>120
        assert '\ufffd' not in t
        pages.append({'page':i,'characters':len(t),'first_lines':t.splitlines()[:3]})
    first=reader.pages[0].extract_text()
    assert '目录' in first and '实验目的与数据' in first and '参考文献' in first;checks+=1
    images=convert_from_path(str(pdf),dpi=140,output_folder=str(BUILD/'render'),fmt='png',
                             output_file='revised',paths_only=True,poppler_path=poppler,thread_count=3)
    report={'status':'structural_checks_passed_pending_visual_review','checks':checks,
            'tables':len(actual),'references':len(refs),'figures':len(figures),
            'native_math_equations':len(xml.findall('.//m:oMath',ns)),
            'pages':pages,'page_images':[str(p) for p in images],
            'renderer':'Native Microsoft Word COM ExportAsFixedFormat; bundled Poppler rasterization',
            'renderer_fallback':'Packaged render_docx.py attempted; LibreOffice missing on Windows. See renderer_initial.log.'}
    (BUILD/'report_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','checks','tables','references','figures','native_math_equations')},ensure_ascii=False))
    for p in pages:print(p['page'],p['characters'],' / '.join(p['first_lines']))

if __name__=='__main__':main()
