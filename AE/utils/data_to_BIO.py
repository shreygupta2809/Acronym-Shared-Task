import sys
import re
import json

with open(sys.argv[1], "r") as f:
    data = json.load(f)

scidr_format_out = []

for sample in data:
    id, text, sf_ptrs, lf_ptrs = [
        sample[key] for key in ["ID", "text", "acronyms", "long-forms"]
    ]
    ptrs = sorted(
        [[sf, "S"] for sf in sf_ptrs] + [[lf, "L"] for lf in lf_ptrs],
        key=lambda x: x[0][0],
    )

    text_len = len(text)

    ftext = []
    ftags = []

    read_idx = 0
    ptrs_idx = 0
    while ptrs_idx < len(ptrs):
        span, typ = ptrs[ptrs_idx]
        if span[0] == read_idx:  # ____form mode
            ftext.append(text[span[0] : span[1]])
            ftags.append(typ)
            ptrs_idx += 1
            read_idx = span[1]
        else:  # text mode
            ftext.append(text[read_idx : span[0]])
            ftags.append("T")  # non-long-nor-short-form-text
            read_idx = span[0]

    if read_idx != text_len:
        ftext.append(text[read_idx:])
        ftags.append("T")  # non-long-nor-short-form-text
        read_idx = text_len

    rtext = []
    rtags = []
    for chunk, mode in zip(ftext, ftags):
        if mode == "T":
            chunk_toks = [
                tok
                for tok in re.sub(
                    "\s{2,}",
                    " ",
                    re.sub(
                        "\u0000",
                        " ",
                        re.sub(
                            "([^a-zA-Z0-9])",
                            r" \1 ",
                            chunk.encode("ascii", errors="ignore").decode().strip(),
                        ),
                    ),
                ).split(" ")
                if tok != ""
            ]
            if len(chunk_toks) > 0:
                rtext.extend(chunk_toks)
                rtags.extend(["O" for _ in range(len(chunk_toks))])
        elif mode == "L":
            chunk_toks = [tok for tok in chunk.strip().split(" ") if tok != ""]
            assert len(chunk_toks) > 0
            rtext.extend(chunk_toks)
            rtags.extend(["B-long"] + ["I-long" for _ in range(len(chunk_toks) - 1)])
        elif mode == "S":
            chunk_toks = [chunk.strip()]
            assert len(chunk_toks) > 0
            rtext.extend(chunk_toks)
            rtags.extend(["B-short"])

    scidr_format_out.append({"id": id, "tokens": rtext, "labels": rtags})

file_list = sys.argv[1].split("/")
dir = "/".join(file_list[:-1])

with open(f"{dir}/scidr_{file_list[-1]}", "w") as f:
    json.dump(scidr_format_out, f, indent=2)
