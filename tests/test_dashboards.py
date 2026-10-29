import json
from pathlib import Path
import re

import pytest
from prometheus_client import REGISTRY

import ollama_exporter as exporter


ROOT = Path(__file__).resolve().parents[1]


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


@pytest.mark.parametrize("filename", ["dashboard.json", "dashboard_custom.json"])
def test_dashboard_datasources_and_metrics(filename):
    dashboard = json.loads((ROOT / filename).read_text())
    assert dashboard["id"] is None
    variables = dashboard["templating"]["list"]
    datasource = next(item for item in variables if item["name"] == "DS_PROMETHEUS")
    assert datasource["type"] == "datasource"
    assert datasource["query"] == "prometheus"
    assert not datasource.get("current")
    metric_names = {sample.name for metric in REGISTRY.collect() for sample in metric.samples}
    # Instantiate every labelled metric once to discover its exported samples.
    for value in vars(exporter).values():
        if hasattr(value, "labels") and getattr(value, "_labelnames", ()) == ("model",):
            value.labels(model="dashboard-validation")
    metric_names.update(sample.name for metric in REGISTRY.collect() for sample in metric.samples)
    for item in objects(dashboard):
        source = item.get("datasource", {})
        if isinstance(source, dict) and source.get("type") == "prometheus":
            assert source["uid"] == "${DS_PROMETHEUS}"
        if "expr" in item:
            referenced = set(re.findall(r"\bollama_[a-z_]+\b", item["expr"]))
            assert referenced, item["expr"]
            assert referenced <= metric_names, referenced - metric_names
            if "ollama_requests_total" in referenced:
                assert "rate(" in item["expr"] or "increase(" in item["expr"]
