"""Fill the ICAAD 2026 Word template with paper content.

The template's own paragraphs are used as prototypes (title, authors,
affiliations, section headings, body text, captions, tables, references), so
the output keeps the conference formatting exactly. All guidance text and
example content from the template is removed.
"""
import copy
import re

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _text(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def _style(el):
    s = el.find(f"{W}pPr/{W}pStyle")
    return s.get(qn("w:val")) if s is not None else None


class PaperBuilder:
    def __init__(self, template):
        self.doc = Document(template)
        self.body = self.doc.element.body
        items = [el for el in self.body if el.tag in (W + "p", W + "tbl")]
        find = lambda pred: next(el for el in items if pred(el))  # noqa: E731
        self.proto = {
            "title": find(lambda e: _style(e) == "Balk1"),
            "authors": find(lambda e: _style(e) == "Yazarlar"),
            "affil": find(lambda e: _style(e) == "Affiliation"),
            "abstract_h": find(lambda e: _text(e).strip() == "ABSTRACT"),
            "keywords": find(lambda e: _text(e).startswith("Keywords")),
            "h1": find(lambda e: _text(e).startswith("1. INTRODUCTION")),
            "h2": find(lambda e: _text(e).startswith("1.1. Subsection")),
            "body": find(lambda e: _text(e).startswith("where n is")),
            "bullet": find(lambda e: _style(e) == "ListeMaddemi"),
            "numbered": find(lambda e: _style(e) == "ListeNumaras"),
            "caption": find(lambda e: _style(e) == "ResimYazs"),
            "figure": find(lambda e: e.tag == W + "p" and e.find(f".//{W}drawing") is not None),
            "eq_table": find(lambda e: e.tag == W + "tbl" and "RMSE" in _text(e)),
            "data_table": find(lambda e: e.tag == W + "tbl" and "Method" in _text(e)),
            "reference": find(lambda e: _style(e) == "Reference"),
            "compact": find(lambda e: _style(e) == "CompactNote"),
        }
        self.proto = {k: copy.deepcopy(v) for k, v in self.proto.items()}
        self.sect = self.body.find(W + "sectPr")
        for el in items:
            self.body.remove(el)
        # drop the example image relationship target from the output later
        self.out = []

    # ---------------- low level ----------------
    def _para(self, proto, runs):
        """Copy a prototype paragraph and replace its runs.

        runs: list of (text, fmt) where fmt is a set of {"b", "i", "sup", "sub"}.
        The run properties of the prototype's last run are used as the base,
        except for runs flagged "b0" which use the first run (useful for the
        bold label + plain text pattern of "Keywords:" and captions).
        """
        p = copy.deepcopy(proto)
        rs = p.findall(W + "r")
        base_first = copy.deepcopy(rs[0].find(W + "rPr")) if rs and rs[0].find(W + "rPr") is not None else None
        base_last = copy.deepcopy(rs[-1].find(W + "rPr")) if rs and rs[-1].find(W + "rPr") is not None else None
        for child in list(p):
            if child.tag != W + "pPr":
                p.remove(child)
        for text, fmt in runs:
            r = p.makeelement(W + "r", {})
            base = base_first if "first" in fmt else base_last
            rpr = copy.deepcopy(base) if base is not None else p.makeelement(W + "rPr", {})
            if "b" in fmt and rpr.find(W + "b") is None:
                rpr.append(rpr.makeelement(W + "b", {}))
            if "i" in fmt:
                rpr.append(rpr.makeelement(W + "i", {}))
            if "sup" in fmt or "sub" in fmt:
                va = rpr.makeelement(W + "vertAlign", {qn("w:val"): "superscript" if "sup" in fmt else "subscript"})
                rpr.append(va)
            if len(rpr):
                r.append(rpr)
            t = r.makeelement(W + "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
            t.text = text
            r.append(t)
            p.append(r)
        self.out.append(p)
        return p

    @staticmethod
    def parse_inline(text, fmt=()):
        """Very small markup: *italic*, ^{sup}, _{sub}."""
        runs = []
        for tok in re.split(r"(\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})", text):
            if not tok:
                continue
            if tok.startswith("*"):
                runs.append((tok[1:-1], set(fmt) | {"i"}))
            elif tok.startswith("^{"):
                runs.append((tok[2:-1], set(fmt) | {"sup"}))
            elif tok.startswith("_{"):
                runs.append((tok[2:-1], set(fmt) | {"sub"}))
            else:
                runs.append((tok, set(fmt)))
        return runs

    # ---------------- blocks ----------------
    def title(self, text):
        self._para(self.proto["title"], [(text, set())])

    def authors(self, text):
        self._para(self.proto["authors"], self.parse_inline(text))

    def affiliation(self, text):
        self._para(self.proto["affil"], self.parse_inline(text))

    def heading(self, text, level=1):
        self._para(self.proto["h1" if level == 1 else "h2"], [(text, set())])

    def abstract(self, text, keywords):
        self._para(self.proto["abstract_h"], [("ABSTRACT", set())])
        self._para(self.proto["body"], self.parse_inline(text))
        self._para(self.proto["keywords"], [("Keywords: ", {"first"}), (keywords, set())])
        self._para(self.proto["compact"], [])

    def paragraph(self, text):
        self._para(self.proto["body"], self.parse_inline(text))

    def bullets(self, items):
        for it in items:
            self._para(self.proto["bullet"], self.parse_inline(it))

    def numbered(self, items):
        for it in items:
            self._para(self.proto["numbered"], self.parse_inline(it))

    def caption(self, label, text):
        self._para(self.proto["caption"], [(label + " ", {"first"}), (text, set())])

    def spacer(self):
        self._para(self.proto["compact"], [])

    def equation(self, omml_inner, number):
        tbl = copy.deepcopy(self.proto["eq_table"])
        cells = tbl.findall(f".//{W}tc")
        mid = cells[1].find(W + "p")
        for child in list(mid):
            if child.tag != W + "pPr":
                mid.remove(child)
        from lxml import etree
        math = etree.fromstring(f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W[1:-1]}">{omml_inner}</m:oMath>')
        mid.append(math)
        num_p = cells[2].find(W + "p")
        num_p.findall(f".//{W}t")[0].text = f"({number})"
        self.out.append(tbl)

    def table(self, rows, widths_cm, font_half_points=16, bold_first_col=False):
        """rows[0] is the header. widths in cm."""
        tbl = copy.deepcopy(self.proto["data_table"])
        trs = tbl.findall(W + "tr")
        head_tr, body_tr = trs[0], trs[1]
        for tr in trs:
            tbl.remove(tr)
        grid = tbl.find(W + "tblGrid")
        for g in list(grid):
            grid.remove(g)
        tw = [int(w * 567) for w in widths_cm]
        for w in tw:
            grid.append(grid.makeelement(W + "gridCol", {qn("w:w"): str(w)}))
        for ri, row in enumerate(rows):
            proto_tr = head_tr if ri == 0 else body_tr
            tr = copy.deepcopy(proto_tr)
            tcs = tr.findall(W + "tc")
            proto_tc = tcs[0] if ri == 0 else tcs[-1]
            for tc in tcs:
                tr.remove(tc)
            for ci, val in enumerate(row):
                tc = copy.deepcopy(proto_tc)
                tc.find(f"{W}tcPr/{W}tcW").set(qn("w:w"), str(tw[ci]))
                p = tc.find(W + "p")
                r = p.find(W + "r")
                for extra in p.findall(W + "r")[1:]:
                    p.remove(extra)
                rpr = r.find(W + "rPr")
                if rpr is None:
                    rpr = r.makeelement(W + "rPr", {})
                    r.insert(0, rpr)
                sz = rpr.find(W + "sz")
                if sz is None:
                    sz = rpr.makeelement(W + "sz", {})
                    rpr.append(sz)
                sz.set(qn("w:val"), str(font_half_points))
                if ri > 0 and ci == 0 and bold_first_col and rpr.find(W + "b") is None:
                    rpr.insert(0, rpr.makeelement(W + "b", {}))
                if ri > 0 and ci == 0:
                    jc = p.find(f"{W}pPr/{W}jc")
                    if jc is not None:
                        jc.set(qn("w:val"), "left")
                r.find(W + "t").text = str(val)
                tr.append(tc)
            tbl.append(tr)
        self.out.append(tbl)

    def figure(self, path, width_cm):
        p = copy.deepcopy(self.proto["figure"])
        for child in list(p):
            if child.tag != W + "pPr":
                p.remove(child)
        self.out.append(p)
        # add the picture through python-docx so the relationship is created
        from docx.text.paragraph import Paragraph
        Paragraph(p, self.doc._body).add_run().add_picture(str(path), width=Cm(width_cm))

    def references(self, refs):
        for r in refs:
            self._para(self.proto["reference"], self.parse_inline(r))

    def save(self, path):
        for el in self.out:
            self.sect.addprevious(el)
        # remove the template's example image relationships that are now unused
        part = self.doc.part
        used = set(re.findall(r'r:embed="(rId\d+)"', self.doc.element.xml))
        for rid, rel in list(part.rels.items()):
            if rel.reltype.endswith("/image") and rid not in used:
                del part.rels[rid]
        self.doc.save(path)
