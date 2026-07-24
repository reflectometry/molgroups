## Publishing to PyPi.org
The workflow at .github/workflows/publish_pypi.yml is set up to publish a wheel for molgroups to PyPi.org

### To make a new version
1. bump the version in pyproject.toml
2. commit and push the change
3. in "Actions", select the publish workflow and run it manually (using default branch)
