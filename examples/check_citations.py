"""无模型演示：仅检查自造法条夹具，不用于法律咨询。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from legal_check import cn2int, verify_citations

# 为软件行为自造的条文，不是真实法条。
db = {'演示法': {(234, 0): '用户提交材料后，应当保留原始记录和完整附件，以便核对版本及其来源。',
                (234, 1): '新增附件应当记录独立编号，并与原始附件分别保存，以便明确识别。'}}
cases = [
    ('《演示法》第234条规定，用户提交材料后，应当保留原始记录和完整附件，以便核对版本及其来源。', 'ok'),
    ('《演示法》第234条之一规定，新增附件应当记录独立编号，并与原始附件分别保存，以便明确识别。', 'ok'),
    ('《演示法》第9999条规定，所有申请都会自动获批。', 'out_of_range'),
]
assert cn2int('二百三十四') == 234
for text, expected in cases:
    result = verify_citations(text, db)
    assert len(result) == 1 and result[0]['verdict'] == expected, result
    print(json.dumps({'synthetic': True, 'text': text, 'verdict': result[0]['verdict']}, ensure_ascii=False))
