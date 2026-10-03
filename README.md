# fuzzy-dedup

跨导入批次的精确模糊去重库（纯 Python，无外部依赖）。只处理进程内顺序调用，
记录只增不改；只输出候选对，不做聚类或实体合并。

## 匹配语义

- 创建 `Matcher(fields=[...])` 时指定参与匹配的字段，之后不可更改。
- 每条记录有全局唯一字符串 `id`；字段值只含字符串、`None` 或缺失。
- 字符串依次经 NFKC 规范化与 casefold，再按 `str.isalnum()` 为真的连续字符取词；
  null / 缺失 / 空字符串不产生词；同字段重复词只计一次；同一词出现在不同字段
  视为不同元素。记录的词集合 = 所有参与字段元素的并集。
- 相似度 = |交集| / |并集|。`|交集| * 100 >= |并集| * 95` 即报告（含恰好 95%）；
  任一集合为空则不匹配。

## API

```python
from fuzzy_dedup import Matcher

m = Matcher(fields=["name", "addr"])
m.import_batch("b1", [{"id": "r1", "name": "...", "addr": "..."}, ...])  # 每批 <= 5000
pairs = m.find_pairs("b1")  # [('r1', 'r7'), ...]，至少一端属于 b1 的全部匹配对
```

`find_pairs` 可查询任何已导入批次；后续导入后再次查询旧批次，会包含它与
新记录之间的匹配。

## 运行全部测试

```bash
python3 -m pytest tests/ -v -s
```

（规模测试也可以单独跑：`python3 tests/scale_benchmark.py`，会打印实际
CPU 耗时与峰值 RSS。）

## 候选筛选为什么不会漏掉满足阈值的记录对

采用前缀过滤（prefix filtering）+ 精确验证：

- 每条记录入库时，把它的 n 个元素按**当时**的全局词频升序排序，取前
  `p(n) = n - ceil(0.95n) + 1` 个（“前缀”）放入倒排索引；新记录用**自己的
  全部元素**去探测索引，命中的记录成为候选，再逐个精确计算 Jaccard 判定。
- 不漏配的证明：设匹配对 (x, y)，|x| = n ≤ |y| = m。Jaccard ≥ 0.95 要求
  并集 ≤ 交集/0.95，且 m ≤ n/0.95（否则相似度至多为 n/m < 0.95）。于是
  交集大小 ≥ 0.95·(n+m)/1.95 ≥ 0.95·m > ceil(0.95·m) − 1 = y 的后缀长度，
  即至少有一个公共元素落在 y 已索引的前缀中；x 探测时遍历自己的全部元素，
  必然经过该元素从而召回 y。
- 该证明只依赖 y 的前缀**长度**，与排序所用的词频无关：词频顺序纯粹是
  性能启发式（稀有词进前缀、高频词进后缀，使倒排表极短）。因此导入顺序、
  批次划分、历史词频变化都不会让任何满足阈值的对消失；也不存在近似召回、
  候选截断或高频词删除。

## 布局

- `fuzzy_dedup/tokenizer.py` — NFKC + casefold + isalnum 取词
- `fuzzy_dedup/matcher.py` — 前缀过滤索引、精确验证、批次化查询
- `tests/reference.py` — 独立 O(n²) 穷举参照实现
- `tests/test_correctness.py` — 随机对照、导入顺序/批次划分不变性、边界语义
- `tests/test_frequency_shift.py` — 词频剧变后旧批次仍能命中（含纯高频词重合）
- `tests/scale_benchmark.py` / `tests/test_scale.py` — 10 万条规模测试
