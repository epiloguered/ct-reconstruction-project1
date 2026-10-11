"""将核查过的文献映射为报告编号；软件文档只标访问日期，不推断发表年份。"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'outputs/report_build'
references=json.loads((folder/'references.json').read_text(encoding='utf-8'))
rows=[]
for ref in sorted((r for r in references if r.get('report_number')),key=lambda r:r['report_number']):
    authors='; '.join(ref['authors'])
    url='https://doi.org/'+ref['doi'] if ref.get('doi') else ref['url']
    venue=ref['venue']+(', '+str(ref['year']) if ref['type']=='paper' else '')
    text=f"{authors}. {ref['title']}. {venue}. {url}"
    if ref['type']!='paper':text+=' (accessed 2026-10-11)'
    rows.append(dict(number=ref['report_number'],text=text,url=url,key=ref['key']))
assert [r['number'] for r in rows]==list(range(1,17))
(folder/'report_bibliography.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('16 checked references mapped; software documentation publication years omitted.')
