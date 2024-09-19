import re
from collections import defaultdict

import onnx
import duckdb
import pandas as pd
import onnx_graphsurgeon as gs
from tqdm import tqdm


def extract_layers(input_str, debug=False):
    # Define the regex patterns for different types of inputs
    input_str = input_str.replace("\u001e", " + ")
    input_str = input_str.replace("\u001f", " + ")
    input_str = input_str.replace("||", " + ")
    input_str = input_str.replace(", ", " + ")

    def foreign(captures):
        captures = ["Foreign" + capture.replace(" + ", " & ") for capture in captures]
        return captures

    patterns = [
        (None, r"[ \n\t\r]*\{[ \n\t\r]*\"Name\":[ \n\t\r]*\"({capture})\"[ \n\t\r]*\}[ \n\t\r]*", r".*", "Unwrap Name Json"),
        (foreign, r"\{ForeignNode({capture})\}", r"\[.*\]", "Unwrap ForeignNode"),
        (None, r"Reformatting CopyNode for (?:Input|Output) Tensor [0-9] to ({capture})", r".*", "Unwrap Reformatting Node"),
        (None, r"\[ONNX Layer: ({capture})\]", r"[a-zA-Z0-9\.\_\/]+", "Unwrap ONNX Layer"),
        (None, r"PWN\(({capture})\)", r"[a-zA-Z0-9\.\_\/\ \+]+", "Unwrap PWN Layer"),
        (None, r"\([^)]+\) \[[^]]+\]_({capture})", r"[a-zA-Z0-9\.\_\/]+", "Unwrap Unnamed Layer"),
        (None, r"\([^)]+\) \[[^]]+\]", None, "Unwrap Unnamed Layer w/o Capture"),
        (None, r"\(({capture})\)", r"[a-zA-Z0-9\.\_\/]+", "Unwrap (Node)"),
        (None, r"\([\{\}\[\] \n\t\r]*\)", None, "Unwrap Brackets()"),
        (None, r"\[[\(\)\{\} \n\t\r]*\]", None, "Unwrap Brackets[]"),
        (None, r"\{[\[\]\(\) \n\t\r]*\}", None, "Unwrap Brackets{}"),
        # (None, r"({capture})", r"([a-zA-Z0-9\.\_\/]+)", "capture ALL"),
    ]

    string = input_str
    max_iter = 10
    for _ in range(max_iter):
        for prio, (func, pattern, capture_pat, desc) in enumerate(patterns):
            if capture_pat is None:
                match_pattern = pattern
            else:
                match_pattern = pattern.replace(r"{capture}", r"?:" + capture_pat)
            matches = re.findall(match_pattern, string)
            is_matched = len(matches) > 0
            if capture_pat is not None:
                capture_pattern = pattern.replace(r"{capture}", capture_pat)
                captures = re.findall(capture_pattern, string)
            else:
                capture_pattern = None
                captures = [''] * len(matches)

            if func is not None:
                captures = func(captures)

            next = string
            for match, capture in zip(matches, captures):
                next = next.replace(match, capture)

            if debug:
                print(f"{desc.ljust(40)}: [ {str(is_matched):<6}] {string} -> {next}")
            string = next
            if is_matched:
                break
        else:
            break
    else:
        raise Exception("Max Iteration Reached")

    names = set()
    for name in string.split("+"):
        name = name.strip()
        if len(name) != 0:
            names.add(name)
    names = list(names)
    return names


def query(cur, sql):
    exe = cur.execute(sql)
    column_names = [description[0] for description in exe.description]
    return pd.DataFrame(exe.fetchall(), columns=column_names)


def insert_nodes_table(cur):
    check = query(cur, "SELECT name FROM sqlite_master WHERE type='table' AND name='NODES'")
    if not check.empty:
        return
    text_df = query(cur, r"""
        SELECT
            string_ids.id as id,
            string_ids.value as text
        FROM StringIds AS string_ids
        JOIN NVTX_EVENTS AS nvtx
            ON nvtx.textid = string_ids.id
        JOIN ENUM_NSYS_EVENT_TYPE AS event_type
            ON nvtx.eventType = event_type.id
        WHERE event_type.name = 'NvtxPushPopRange'
        GROUP BY string_ids.id, string_ids.value
    """)

    nodes_data = []
    for row in text_df.itertuples():
        nodes = extract_layers(row.text, debug=False)
        nodes_data.append({"id": row.id, "nodes": "+c+a+t+".join(nodes)})

    nodes_df = pd.DataFrame(nodes_data)
    cur.execute("CREATE TABLE NODES AS SELECT * FROM nodes_df")


def insert_onnx_graph_table(cur):
    check = query(cur, "SELECT name FROM sqlite_master WHERE type='table' AND name='ONNX_GRAPH'")
    if not check.empty:
        return
    onnx_model = onnx.load("simplified_onnx/unetxl.onnx")
    onnx.checker.check_model("simplified_onnx/unetxl.onnx")
    graph = gs.import_onnx(onnx_model)
    nodes = graph.nodes
    graph_data = []
    for node in nodes:
        node_data = {}
        node_data["name"] = str(node.name)
        node_data["op"] = str(node.op)
        node_data["inputs"] = str(node.inputs)
        node_data["outputs"] = str(node.outputs)
        node_data["attrs"] = str(node.attrs)
        graph_data.append(node_data)
    graph_df = pd.DataFrame(graph_data)
    cur.execute("CREATE TABLE ONNX_GRAPH AS SELECT * FROM graph_df")


def load_cur():
    cursor = duckdb.connect()
    cursor.execute("""
        INSTALL sqlite;
        LOAD sqlite;
        ATTACH 'report2.sqlite' AS report (TYPE sqlite);
        USE report;
    """)
    return cursor


def analyze(cur):
    check = query(cur, "SELECT name FROM sqlite_master WHERE type='table' AND name='ANALYSIS'")
    if not check.empty:
        return query(cur, "SELECT * FROM ANALYSIS")
    analysis_df = query(cur, '''
        WITH cupti_kernel AS (
        SELECT
            cupti_kernel.start as start_time,
            cupti_kernel.end as end_time,
            cupti_kernel.correlationId,
            string_ids.value AS cupti_kernel_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS cupti_kernel
        JOIN StringIds AS string_ids
            ON string_ids.id = cupti_kernel.demangledName
        )
        , nvtx AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY nvtx.start) row_num,
                cupti_kernel.start_time as start_time,
                cupti_kernel.end_time as end_time,
                string_ids.id as string_id,
                STRING_SPLIT(Nodes.nodes, '+c+a+t+') as nodes,
                cupti_kernel.cupti_kernel_name AS cupti_kernel_name,
            FROM NVTX_EVENTS AS nvtx
            JOIN ENUM_NSYS_EVENT_TYPE AS event_type
                ON nvtx.eventType = event_type.id
            JOIN StringIds AS string_ids
                ON nvtx.textid = string_ids.id
            JOIN NODES AS Nodes
                ON nvtx.textid = Nodes.id
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS cupti_runtime
                ON nvtx.start <= cupti_runtime.start AND cupti_runtime.end <= nvtx.end
            JOIN cupti_kernel AS cupti_kernel
                ON cupti_runtime.correlationId = cupti_kernel.correlationId
            WHERE event_type.name = 'NvtxPushPopRange'
        )
        , nvtx_unnested AS (
            SELECT
                row_num, start_time, end_time, node['nodes'] AS node, cupti_kernel_name
            FROM nvtx
            CROSS JOIN UNNEST(nvtx.nodes) AS node
        )
        SELECT *
        FROM ONNX_GRAPH AS onnx_graph
        JOIN nvtx_unnested AS nvtx_unnested
            ON nvtx_unnested.node = onnx_graph.name
        ORDER BY nvtx_unnested.row_num
    ''')
    cur.execute("CREATE TABLE ANALYSIS AS SELECT * FROM analysis_df")
    return analysis_df


if __name__ == "__main__":
    cur = load_cur()
    insert_nodes_table(cur)
    insert_onnx_graph_table(cur)
    df = analyze(cur)
    print(df)
    is_conv = lambda x: "cudnn" in x or "conv" in x or ("nchw" in x and "gemm" in x)
    is_mha = lambda x: "mha" in x
    is_gemm = lambda x: "gemm" in x and not is_conv(x) and not is_mha(x)

    df["is_conv"] = df.cupti_kernel_name.map(is_conv)
    df["is_mha"] = df.cupti_kernel_name.map(is_mha)
    df["is_gemm"] = df.cupti_kernel_name.map(is_gemm)
    print(df["is_gemm"].sum(), df["is_conv"].sum(), df["is_mha"].sum())
    df["duration"] = df.end_time - df.start_time
    df["is_compute"] = df.is_gemm | df.is_conv | df.is_mha
    compute_node_df = df[df["is_compute"]].groupby("name").agg(
        {
            "row_num": "first",
            "attrs": "count",
            "duration": "mean",
            "cupti_kernel_name": "first",
            "is_conv": "first",
            "is_mha": "first",
            "is_gemm": "first",
            "inputs": "first",
            "outputs": "first"
        }
    ).reset_index()
    assert len(compute_node_df["attrs"].unique()) == 1, "The number of occurrences of the all nodes should be the same"
    compute_node_df = compute_node_df.sort_values("row_num", ascending=False)
    grouped_compute_node_df = compute_node_df.groupby("row_num").agg(
        {
            "name": lambda x: " + ".join(x.values),
            "duration": "first",
            "cupti_kernel_name": "first",
            "is_conv": "first",
            "is_mha": "first",
            "is_gemm": "first",
            "inputs": "first",
            "outputs": "first"
        }
    ).sort_values("duration", ascending=False).reset_index()
    print(
        grouped_compute_node_df[grouped_compute_node_df.is_conv].duration.sum() / grouped_compute_node_df.duration.sum(),
        grouped_compute_node_df[grouped_compute_node_df.is_mha].duration.sum() / grouped_compute_node_df.duration.sum(),
        grouped_compute_node_df[grouped_compute_node_df.is_gemm].duration.sum() / grouped_compute_node_df.duration.sum()
    )
    grouped_compute_node_df.to_csv("compute_node.csv", index=False, sep="\t")

    graph_df = pd.read_csv("sd_flops.csv", sep="\t")
    graph_nodes = {v: 0 for v in graph_df["name"].values}
    not_graph_nodes = defaultdict(int)
    for row in grouped_compute_node_df.itertuples():
        node_names = row.name.split(" + ")
        is_counted = False
        for _node_name in node_names:
            node_name = ".".join(_node_name.strip().split("/")[1:-1])
            if node_name == "down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_k":
                import pdb; pdb.set_trace()
            if node_name in graph_nodes:
                if not is_counted:
                    graph_nodes[node_name] += row.duration
                else:
                    print(f"Node {node_name} already found in graph (row: {row})")
                is_counted = True
            elif node_name.endswith("input_quantizer") or node_name.endswith("weight_quantizer"):
                continue
            else:
                not_graph_nodes[node_name] += row.duration
                print(f"Node {node_name} ({_node_name}, {row.cupti_kernel_name}) not found in graph")
    print(f"Graph Nodes: {len(graph_nodes)}")
    print(f"Not Graph Nodes: {len(not_graph_nodes)}")
    print(f"Graph Nodes Duration: {sum(graph_nodes.values())}")
    print(f"Not Graph Nodes Duration: {sum(not_graph_nodes.values())}")
    print(f"Graph Nodes Duration: {sum(graph_nodes.values()) / (sum(graph_nodes.values()) + sum(not_graph_nodes.values()))}")
    graph_df["trt_duration"] = graph_df.name.map(lambda x: graph_nodes.get(x, 0))
    graph_df.to_csv("sd_flops_trt.csv", index=False, sep="\t")
    cur.close()
