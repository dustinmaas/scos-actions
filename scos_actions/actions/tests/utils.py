from sigmf.validate import validate as sigmf_validate

SENSOR_DEFINITION = {
    "id": "",
    "sensor_spec": {"id": "", "model": "greyhound"},
    "antenna": {"antenna_spec": {"id": "", "model": "L-com HG3512UP-NF"}},
    "signal_analyzer": {"sigan_spec": {"id": "", "model": "Ettus USRP B210"}},
    "computer_spec": {"id": "", "model": "Intel NUC"},
}


def check_metadata_fields(
    metadata, action, entry_name, action_name, task_id, recording=None
):
    assert sigmf_validate(metadata)
    # schema_validate(sigmf_metadata, schema)
    assert "ntia-scos:action" in metadata["global"]
    assert metadata["global"]["ntia-scos:action"]["name"] == action_name
    assert metadata["global"]["ntia-scos:action"]["description"] == action.description
    assert metadata["global"]["ntia-scos:action"]["summary"] == action.summary
    assert "ntia-scos:schedule" in metadata["global"]
    assert metadata["global"]["ntia-scos:schedule"]["name"] == entry_name
    assert "ntia-scos:task" in metadata["global"]
    assert metadata["global"]["ntia-scos:task"] == task_id
    if recording:
        assert "ntia-scos:recording" in metadata["global"]
        assert metadata["global"]["ntia-scos:recording"] == recording
    else:
        assert "ntia-scos:recording" not in metadata["global"]

    assert "ntia-core:classification" in metadata["global"]
    assert metadata["global"]["ntia-core:classification"] == "UNCLASSIFIED"
    assert len(metadata["captures"]) >= 1
