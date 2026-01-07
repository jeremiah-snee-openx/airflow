# Improvements - concept refinement

## Naming
mirror the Google provider already names the GKE operator module:
- airflow.providers.google.cloud.decorators.kubernetes_engine
- airflow.providers.google.cloud.decorators.kubernetes_engine_cmd


- add provider config options (in airflow.cfg / environment variables) rather than Airflow Variables.

To make this “official upstream quality,” moving your *cluster-defaults* off Airflow **Variables** and into **provider config** is exactly the right direction: it makes defaults deploy-time concerns (Helm chart / env vars / airflow.cfg), not runtime mutable state.

## What “provider config options” means in Airflow

Providers can declare configuration sections + options in **`provider.yaml`** under `config`. The schema requires each config section to have a `description` and an `options` map; and each option must include `description`, `version_added`, `type`, `example`, and `default`. ([Apache Git Repositories][1])

Also, options can be marked `sensitive: true`, which enables `__SECRET` / `__CMD` env var variants (if you ever add something secret—though for GKE cluster identifiers you generally should not). ([Apache Git Repositories][1])

## Naming conventions to follow (validated)

A real upstream example is the JDBC provider: docs explicitly reference config in the **`providers.jdbc`** section, and show the environment variable form `AIRFLOW__PROVIDERS_JDBC__...` (note the dot → underscore transform in env var “section”). ([Apache Airflow][2])

So for Google+GKE decorators, the clean upstream pattern is:

* **airflow.cfg section**: `[providers.google.gke]`
* **env var section token**: `PROVIDERS_GOOGLE_GKE`
* **env vars**: `AIRFLOW__PROVIDERS_GOOGLE_GKE__<OPTION_NAME_IN_UPPERCASE>`

## What I would name the section and options

Because the Google provider is huge, don’t dump these into a generic `[providers.google]` section. Make it tightly scoped:

### Section

* **`providers.google.gke`**

### Options (cluster defaults for TaskFlow decorators)

These match exactly the fields you’re already resolving (project/location/cluster/namespace/gcp_conn_id), but sourced from config instead of Variables:

* `default_project_id` (string, default: null)
* `default_location` (string, default: null)
* `default_cluster_name` (string, default: null)
* `default_namespace` (string, default: `"default"`)
* `default_gcp_conn_id` (string, default: null)
  *Reasoning: don’t assert a global default here—let the operator/provider default apply if unset.*
* `default_random_name_suffix` (boolean, default: true)
  *Nice to centralize because it affects naming determinism across decorators.*

If you later add “opinionated defaults” for images, labels, service account name, etc., put them here too—but keep this first pass strictly about *cluster targeting*.

## `provider.yaml` snippet (drop into Google provider)

This is schema-compliant (all required keys present). `version_added` is intentionally `null` so you don’t lie in-source; you fill it in when you know the provider release version that will ship it. ([Apache Git Repositories][1])

```yaml
config:
  providers.google.gke:
    description: "Defaults used by Google provider GKE TaskFlow decorators (e.g. @task.gke_pod, @task.gke_job)."
    options:
      default_project_id:
        description: "Default GCP project id used when project_id is not provided to the decorator."
        version_added: null
        type: string
        example: "<gcp-project-id>"
        default: null

      default_location:
        description: "Default GKE location/region/zone used when location is not provided to the decorator."
        version_added: null
        type: string
        example: "<gke-location>"
        default: null

      default_cluster_name:
        description: "Default GKE cluster name used when cluster_name is not provided to the decorator."
        version_added: null
        type: string
        example: "<gke-cluster-name>"
        default: null

      default_namespace:
        description: "Default Kubernetes namespace used when namespace is not provided to the decorator."
        version_added: null
        type: string
        example: "default"
        default: "default"

      default_gcp_conn_id:
        description: "Default Airflow connection id for Google credentials used when gcp_conn_id is not provided to the decorator."
        version_added: null
        type: string
        example: "<airflow-connection-id>"
        default: null

      default_random_name_suffix:
        description: "Whether to enable random_name_suffix by default for GKE pod/job decorators."
        version_added: null
        type: boolean
        example: "true"
        default: "true"
```

## Example airflow.cfg + env vars

### airflow.cfg

```ini
[providers.google.gke]
default_project_id = my-prod-project
default_location = us-central1
default_cluster_name = prod-cluster-1
default_namespace = airflow
default_gcp_conn_id = google_cloud_default
default_random_name_suffix = true
```

### Environment variables (same values)

This follows the same documented pattern as `providers.jdbc` (section normalized for env vars). ([Apache Airflow][2])

```bash
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_PROJECT_ID="my-prod-project"
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_LOCATION="us-central1"
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_CLUSTER_NAME="prod-cluster-1"
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_NAMESPACE="airflow"
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_GCP_CONN_ID="google_cloud_default"
export AIRFLOW__PROVIDERS_GOOGLE_GKE__DEFAULT_RANDOM_NAME_SUFFIX="true"
```

## How your decorator should consume it (behavior, not guesses)

For **opinionated-default** variants (`gke_pod`, `gke_pod_cmd`, `gke_job`, `gke_job_cmd`):

Precedence should become:

1. explicit decorator args (what the DAG author passes)
2. provider config (`providers.google.gke`)
3. (only for namespace / random suffix) hard default (`default`, `true`)
4. if still missing for required fields (project/location/cluster): raise `AirflowException`

That is a clean, deterministic contract and mirrors what you were doing with Variables, just in the correct config system.

For **alpha/no-opinion** variants (`*_cmd_alpha`):

* Don’t read provider config at all (or read it only if you explicitly decide alpha should still support config). If you want “no opinionated defaults,” the strictest interpretation is: **only use explicit args**.

---

If you want the next step to be equally “upstream grade,” it’s: update your `resolve_gke_defaults(...)` helper to read from Airflow config (not Variables), and update the exception text to point users at `[providers.google.gke]` / env vars instead of Variable keys.

[1]: https://apache.googlesource.com/airflow/%2B/refs/heads/main/airflow-core/src/airflow/provider.yaml.schema.json "airflow-core/src/airflow/provider.yaml.schema.json - airflow - Git at Google"
[2]: https://airflow.apache.org/docs/apache-airflow-providers-jdbc/stable/connections/jdbc.html?utm_source=chatgpt.com "JDBC connection — apache-airflow-providers-jdbc Documentation"
