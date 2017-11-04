from lxml import html
import requests

page = requests.get('https://eresearch.fidelity.com/eresearch/markets_sectors/sectors/si_performance.jhtml?tab=siperformance')
tree = html.fromstring(page.content)
print(tree.xpath("//a[re:match(text(), '<a href=(.*) tr>')]", namespaces={"re": "http://exslt.org/regular-expressions"}))