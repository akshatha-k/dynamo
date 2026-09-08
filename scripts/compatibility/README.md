<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-version component compatibility

Test a candidate frontend and SGLang worker against the previous two release
lines through an independent HTTP client. The candidate is the code being
validated: for a PR targeting 1.5, the historical release lines are 1.4 and 1.3.
This tests the frontend/worker protocol boundary used during upgrades; it does
not perform a Kubernetes rolling upgrade or measure routing capacity.

## Version matrix

Run all five pairs for both embedding and aggregated Chat Completions:

| Frontend | Worker |
| --- | --- |
| candidate | candidate |
| candidate | N-1 |
| N-1 | candidate |
| candidate | N-2 |
| N-2 | candidate |

The target release line defaults to the checkout's Cargo.toml version and can
be specified with `--release-line`. `releases.json` selects one published patch
per historical minor line (currently 1.4.2 and 1.3.1). Update the manifest as the
candidate advances. Missing historical entries fail rather than reducing
coverage. The resolver requires two previous minors within the same major; a
major transition needs an explicitly reviewed window policy. This samples the
listed patches, not every historical patch. There is no published-only matrix
or separate historical defect reproduction suite.

Both candidate images are mandatory. Kubernetes CI resolves each registry tag
once with `skopeo` and deploys its digest. It retains the tag's runtime version
in `runtimeVersionOverride`, so operator configuration still matches historical
components. The evidence records image references, pinned digests, Pod image IDs,
installed component versions, and immutable model revisions. Init containers
materialize the same pinned model revision at `/model` for both components;
inference runs from `/tmp` and loads models offline. No current-checkout code or
adapter is injected into either component.

CI uses a dedicated vCluster with shared etcd/NATS and a new DGD per scenario.
A one-GPU resource quota and sequential execution bound GPU allocation; each
scenario waits for its Pods to disappear before the next starts. The local
Docker runner instead owns a network, etcd and NATS per scenario, pins images
to local content IDs, and copies model snapshots into its containers.

## Initial contracts

- **Embedding:** Qwen3-Embedding-0.6B; default encoding, explicit `float`, and
  batch inputs. Require finite JSON vectors, expected dimensions, cardinality
  and indices. HTTP errors and strings returned for float requests fail; the
  client does not normalize an incompatible response.
- **Chat Completions:** Qwen2.5-0.5B-Instruct; unary, streaming, `max_tokens=1`,
  and `stop`. Validate content, token limits, finish reasons,
  stream errors and `[DONE]`. Derive a stop string from the unary response's
  prefix and repeat at temperature 0; require an empty stopped response with
  finish reason `stop`. This assumes deterministic greedy output for the same
  request on the same worker, not a particular model-generated phrase.

The stop case depends on a successful unary baseline. If that fails, the
scenario already fails and remaining independent cases still run. There are
no xfails for known protocol incompatibilities. These are API/protocol checks,
not embedding numerical parity or model-quality benchmarks. V1 does not cover
disaggregation, KV-aware routing, tools, multimodal inference, other engines,
or simultaneous mixed worker pools.

## Run locally or on a GPU runner

Requires Linux amd64, one NVIDIA GPU with at least 24 GiB VRAM, a driver
compatible with every selected CUDA image, and Docker with NVIDIA runtime.
The client and daemon must share the network namespace (local Docker or a
DinD sidecar) so published loopback ports are reachable. Remote Docker with a
separate network namespace is unsupported. Allow disk for all runtime images
and models. Log in to private candidate registries before invoking the runner.

```bash
python3 -m venv /tmp/n2-client
/tmp/n2-client/bin/pip install requests==2.32.5 huggingface-hub==0.34.4
/tmp/n2-client/bin/python -m unittest discover -s scripts/compatibility -v
/tmp/n2-client/bin/python scripts/compatibility/runner.py \
  --frontend-image YOUR_FRONTEND_IMAGE --worker-image YOUR_SGLANG_IMAGE \
  --output /tmp/n2-results
```

The output directory must not exist. Add `--plan` to validate/print the matrix
without Docker or GPUs. Use `--config` for another reviewed release/model
manifest. The supplied images must correspond to the target release line.

## CI and evidence

`compatibility-contract-tests.yml` runs CPU harness tests on relevant PRs in
both normal and optimized (`python -O`) mode.
The GPU job lives in `pr.yaml`, on approved `pull-request/N` pushes. It waits
for `frontend-copy-to-acr`, `sglang-copy-to-acr` and the operator build, then
passes their ACR tags to `cross-version-compatibility.yml`. Relevant core,
frontend and SGLang changes trigger the necessary builds and copies. Like other
PR deployment tests, it respects `RUN_DEPLOY_TESTS`; a disabled deployment lane
means no GPU compatibility evidence. Its result participates in
`dynamo-status-check`.

The reusable workflow uses `prod-deploy-tester-v1`, creates a dedicated vCluster
through `setup-dynamo-operator`, and invokes `dynamo-deploy-test` with
`tests/deploy/test_n2_compatibility.py -m k8s -n 0`. The deployment test carries `post_merge` to stay out of the ordinary GPU
pre-merge lane; the dedicated PR workflow explicitly selects it with `-m k8s`.
It also carries `framework_agnostic`: the engine runs in Pods, so the test client
does not need SGLang installed. The failure-injection unit tests are selected by
the SGLang CPU lane (`pre_merge and sglang and gpu_0`).
The test reuses
`ManagedDeployment` for readiness, log capture, port forwarding and teardown.
The runner needs Kubernetes access and skopeo; it needs neither a local GPU nor
a Docker daemon. GPU execution happens in the cluster's Worker Pods. Candidate
images come from ACR, historical release images from NGC. Both must be pullable
by the cluster, and the GPU driver must support all selected CUDA images.

Nightly runs the same ten scenarios with its ACR nightly artifacts. Manual
runs accept `frontend_tag`, `worker_tag`, `operator_tag` and optional
`release_line`. Component tags must start with their semantic runtime version,
as the shared ACR copy workflow's tags do. The job timeout is 300 minutes; a
separate always-run job tears down the vCluster after success, failure or
cancellation. V1 does not change release promotion gates.

Any startup, request, validation or cleanup failure makes the job fail. Later
cases and pairs continue after a scenario failure. Artifacts contain the
plan, per-scenario reports, raw requests/responses or SSE lines, Pod logs,
image IDs, events, installed versions and JUnit results; model weights are excluded. Compare failures with the
candidate/candidate control before attributing them to version skew. A healthy
control is useful evidence but does not by itself rule out an issue specific
to a historical engine or its launch configuration.

Additional scenarios can reuse the version matrix and orchestration with their
own model specification and HTTP assertions. Additional engines need explicit
launchers and supported-version configuration.
