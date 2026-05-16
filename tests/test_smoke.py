from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "wikivir_emotion_topic_analysis.py"
    spec = importlib.util.spec_from_file_location("wikivir_emotion_topic_analysis", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_and_lexicon_smoke(tmp_path: Path):
    mod = load_module()
    conllu = tmp_path / "tiny.conllu"
    conllu.write_text(
        """# newdoc id = d1
# title = Test
# author = Avtor
# sent_id = d1.1
# text = Ljubezen je močna.
1	Ljubezen	ljubezen	NOUN	Ncfsn	_	3	nsubj	_	NER=O
2	je	biti	AUX	Va-r3s-n	_	3	cop	_	NER=O
3	močna	močen	ADJ	Agpfsn	_	0	root	_	NER=O|SpaceAfter=No
4	.	.	PUNCT	Z	_	3	punct	_	NER=O

# sent_id = d1.2
# text = Strah je velik.
1	Strah	strah	NOUN	Ncmsn	_	3	nsubj	_	NER=O
2	je	biti	AUX	Va-r3s-n	_	3	cop	_	NER=O
3	velik	velik	ADJ	Agpmsn	_	0	root	_	NER=O|SpaceAfter=No
4	.	.	PUNCT	Z	_	3	punct	_	NER=O
""",
        encoding="utf-8",
    )
    lex = tmp_path / "sloemolex.tsv"
    lex.write_text(
        "word\tjoy\tfear\tpositive\tnegative\tvalence\tarousal\tdominance\n"
        "ljubezen\t1\t0\t1\t0\t0.9\t0.6\t0.7\n"
        "strah\t0\t1\t0\t1\t0.2\t0.9\t0.4\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = mod.main([
        "--input", str(conllu),
        "--sloemolex", str(lex),
        "--output-dir", str(out),
        "--segment-levels", "document,sentence",
        "--no-lda",
    ])
    assert rc == 0
    assert (out / "tables" / "document_emotion_scores.tsv").exists()
    assert (out / "tables" / "segment_emotion_scores.tsv").exists()
    assert (out / "report.md").exists()


def test_conllu_parser_preserves_doc_metadata(tmp_path: Path):
    mod = load_module()
    conllu = tmp_path / "tiny.conllu"
    conllu.write_text(
        """# newdoc id = doc-x
# genre = roman
# sent_id = doc-x.1
1	A	a	NOUN	_	_	0	root	_	NER=O
""",
        encoding="utf-8",
    )
    artifacts = mod.RunArtifacts()
    docs = mod.parse_conllu(conllu, artifacts)
    assert len(docs) == 1
    assert docs[0].doc_id == "doc-x"
    assert docs[0].metadata["genre"] == "roman"
    assert docs[0].sentences[0].tokens[0].lemma == "a"


def test_parser_accepts_newdoc_prefixed_metadata(tmp_path: Path):
    mod = load_module()
    conllu = tmp_path / "tiny_prefixed.conllu"
    conllu.write_text(
        """# newdoc id = d2
# newdoc title = Naslov
# document.author = Avtorica
# doc_meta = century=19|genre=pesem
# sent_id = d2.1
1	A	a	NOUN	_	_	0	root	_	NER=O
""",
        encoding="utf-8",
    )
    docs = mod.parse_conllu(conllu, mod.RunArtifacts())
    assert docs[0].metadata["title"] == "Naslov"
    assert docs[0].metadata["author"] == "Avtorica"
    assert docs[0].metadata["century"] == "19"
    assert docs[0].metadata["genre"] == "pesem"


def test_transformer_classifier_forwards_batch_size(tmp_path: Path, monkeypatch):
    mod = load_module()
    calls: list[dict[str, object]] = []

    class FakeClassifier:
        def __call__(self, texts, **kwargs):
            calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
            return [
                [
                    {"label": "topic-a", "score": 0.8},
                    {"label": "topic-b", "score": 0.2},
                ]
                for _ in texts
            ]

    fake_transformers = types.ModuleType("transformers")

    def fake_pipeline(task, model=None, tokenizer=None, device=None):
        assert task == "text-classification"
        assert model == "fake/model"
        assert tokenizer == "fake/model"
        assert device == -1
        return FakeClassifier()

    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    segments = [
        mod.Segment(
            segment_id=f"s{i}",
            unit="window",
            doc_id="d1",
            tokens=[],
            text=f"besedilo primer {i}",
            lemma_text=f"besedilo primer {i}",
        )
        for i in range(3)
    ]
    args = types.SimpleNamespace(
        resume=False,
        force=False,
        checkpoint_dir=None,
        progress_every=1000,
        quiet_progress=True,
        transformer_unit="window",
        transformer_min_words=1,
        transformer_max_segments=0,
        transformer_shard_size=10,
        transformer_batch_size=2,
        transformer_device="cpu",
        topic_classifier_model="fake/model",
        transformer_max_chars=1000,
        transformer_max_length=128,
        transformer_top_k=2,
    )
    dirs = mod.ensure_dirs(tmp_path / "out")

    df = mod.run_transformer_topic_classifier(segments, args, dirs, mod.RunArtifacts())

    assert not df.empty
    assert [len(call["texts"]) for call in calls] == [2, 1]
    assert all(call["kwargs"]["batch_size"] == 2 for call in calls)
    assert all(call["kwargs"]["max_length"] == 128 for call in calls)
    assert all(call["kwargs"]["top_k"] == 2 for call in calls)


def test_restore_metadata_from_xml_by_order(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    annotated = tmp_path / "annotated.conllu"
    annotated.write_text(
        """# newdoc id = d-generated
# sent_id = d-generated.1
1	Slovenja	Slovenja	PROPN	_	_	0	root	_	NER=O
""",
        encoding="utf-8",
    )
    xml = tmp_path / "source.xml"
    xml.write_text("""<corpus><doc title="Slovenja" author="Koseski" century="19" genre="pesmi">Besedilo</doc></corpus>""", encoding="utf-8")
    out = tmp_path / "restored.conllu"
    cmd = [
        sys.executable,
        str(root / "scripts" / "restore_wikivir_metadata.py"),
        "--annotated", str(annotated),
        "--source-xml", str(xml),
        "--output", str(out),
        "--match-mode", "by-order",
    ]
    subprocess.run(cmd, check=True)
    restored = out.read_text(encoding="utf-8")
    assert "# title = Slovenja" in restored
    assert "# author = Koseski" in restored
    assert "# century = 19" in restored
    assert "1\tSlovenja" in restored


def test_restore_metadata_from_malformed_xml_recover_mode(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    annotated = tmp_path / "annotated.conllu"
    annotated.write_text(
        """# newdoc id = d1
# sent_id = d1.1
1\tA\ta\tNOUN\t_\t_\t0\troot\t_\tNER=O

# newdoc id = d2
# sent_id = d2.1
1\tB\tb\tNOUN\t_\t_\t0\troot\t_\tNER=O
""",
        encoding="utf-8",
    )
    xml = tmp_path / "bad.xml"
    xml.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<corpus>
<doc title="Slovenja" author="Jovan & Vesel" century="19">Besedilo</doc>
\x01
<doc title='Poezije' author='France Prešeren'>Besedilo</doc>
</corpus>
""",
        encoding="utf-8",
    )
    out = tmp_path / "restored.conllu"
    cmd = [
        sys.executable,
        str(root / "scripts" / "restore_wikivir_metadata.py"),
        "--annotated", str(annotated),
        "--source-xml", str(xml),
        "--xml-parse-mode", "recover",
        "--output", str(out),
        "--match-mode", "by-order",
        "--replace-existing",
    ]
    subprocess.run(cmd, check=True)
    restored = out.read_text(encoding="utf-8")
    assert "# title = Slovenja" in restored
    assert "# author = Jovan & Vesel" in restored
    assert "# title = Poezije" in restored
    assert "# author = France Prešeren" in restored


def load_annotator_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "annotate_wikivir_xml_classla.py"
    spec = importlib.util.spec_from_file_location("annotate_wikivir_xml_classla", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_wikivir_xml_annotator_dry_run_preserves_metadata(tmp_path: Path, capsys):
    mod = load_annotator_module()
    xml = tmp_path / "wikivir.xml"
    xml.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<corpus>
  <doc title="Slovenja" author="Koseski" century="19" genre="pesmi">Slovenja\n\nBesedilo.</doc>
  <doc title="Poezije" author="France Prešeren">Dolgost življenja.</doc>
</corpus>
""",
        encoding="utf-8",
    )
    rc = mod.main([
        "--input", str(xml),
        "--output", str(tmp_path / "out.conllu"),
        "--dry-run",
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert '"documents_seen": 2' in printed
    assert '"title": "Slovenja"' in printed
    assert '"author": "France Prešeren"' in printed


def test_wikivir_xml_annotator_comment_builder():
    mod = load_annotator_module()
    doc = mod.XmlDocument(
        index=1,
        doc_id="wikivir-000001",
        text="Slovenja.",
        metadata={"title": "Slovenja", "author": "Koseski", "century": "19", "genre": "pesmi"},
        raw_attrs={},
        source="/tmp/wikivir.xml",
    )
    sent = mod.ConlluSentence(comments=["# text = Slovenja."], rows=["1\tSlovenja\tSlovenja\tPROPN\t_\t_\t0\troot\t_\tNER=O"])
    comments = mod.sentence_comments(
        document=doc,
        sentence=sent,
        sentence_number=1,
        block_number=1,
        first_sentence_in_doc=True,
        first_sentence_in_block=True,
        block_mode="blanklines",
        metadata_keys=["title", "author", "century", "genre"],
    )
    joined = "\n".join(comments)
    assert "# newdoc id = wikivir-000001" in joined
    assert "# title = Slovenja" in joined
    assert "# author = Koseski" in joined
    assert "# century = 19" in joined
    assert "# genre = pesmi" in joined
    assert "# sent_id = wikivir-000001.s000001" in joined
