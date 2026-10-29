# Releasing

Releases contain source code, the Dockerfile, and both dashboards. GitHub
provides ZIP and tar.gz source archives. We do not publish container images or
installers. The Git tag is the version; there is no separate package version.

## Choose a version

Use stable semantic versions such as `v0.1.0`. Until 1.0, increment the minor
version for incompatible changes and the patch version for compatible fixes.
Release shipped exporter, dependency, container, or dashboard changes when they
are ready for users. Documentation, tests, CI, and repository maintenance alone
do not warrant a software release. Use `docs:`, `ci:`, or `chore:` commit and PR
subjects for that maintenance.

The first release is `v0.1.0`, the runtime baseline covered by the
[live validation record](../docs/validation/2026-10-29.md). Adding this workflow
does not itself constitute a new runtime change.

## Publish

1. Merge the intended changes through a pull request with all required checks
   passing. Do not change an already merged PR to prepare a release.
2. From **Actions > Release > Run workflow**, select `main` and enter the new
   version. The CLI equivalent is:

   ```sh
   gh workflow run release.yml --repo frcooper/ollama-exporter --ref main -f version=v0.1.0
   ```

3. Watch the run. It rejects existing tags, older versions, and releases with
   no shipped changes since the previous release. It reruns all four Python
   versions and the container build against the selected commit, then creates
   that exact tag and publishes a GitHub release with contributor release notes.
4. Check that the release links to the tested commit and offers its source
   archives. GitHub release immutability must remain enabled so published tags
   and assets cannot be replaced. Release descriptions can still be corrected.

The workflow only runs from `main`, and concurrent release runs are serialized.
Ordinary pushes, PRs, and docs merges never publish releases. Only the publishing
job receives `contents: write`; validation keeps the read-only token. No package
registry credentials or image publishing permissions are needed.

If publication fails after creating a tag, inspect the failed run and the tag
before retrying. The workflow refuses to reuse tags. Do not force-move a tag or
delete/recreate a published release; repair the workflow through a new PR and
choose a new version if needed.

Consumers can use a release tag for a stable source pin, or the full commit SHA
for an explicit code identity. Direct Python dependencies are pinned, but
transitive dependencies and the Docker base tag can change between builds.
