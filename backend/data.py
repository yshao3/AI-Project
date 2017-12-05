from lxml import html, etree
import requests

page = requests.get('https://eresearch.fidelity.com/eresearch/markets_sectors/sectors/si_performance.jhtml?tab=siperformance')
tree = html.fromstring(page.content)
# print(tree.xpath("*", namespaces={"re": "http://exslt.org/regular-expressions"}))
print(etree.tostring(tree))