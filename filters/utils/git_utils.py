from subprocess import check_output, CalledProcessError
from typing import Tuple, Optional


def git_info(project_home: str = None) -> Tuple[Optional[str], Optional[str]]:
    """Read the git branch and commit of the project's home folder.
    This is useful to log your branch and commit has for referencing results

    Args:
        project_home: The project home folder
        (usually the location main.py is called from)
    Returns:
        git_branch: Associated git tag/branch name. None if not possible to get.
        git_commit: The git hash, if exists. None if not possible to get.

    """

    git_branch = None
    git_commit = None

    cmd = ["git", "rev-parse", "HEAD"]
    try:
        _hash = check_output(cmd, cwd=project_home)
        git_commit = str(_hash, "utf-8").strip()
    except CalledProcessError:
        print("Not able to get git commit hash: {cmd} @ {project_home}").format(
            cmd=cmd, project_home=project_home
        )

    cmd = "git symbolic-ref -q --short HEAD || git describe --tags --exact-match"
    try:
        _name = check_output(cmd, cwd=project_home, shell=True)
        git_branch = str(_name, "utf-8").strip()
    except CalledProcessError:
        print("Not able to get git branch name: '{cmd}' @ {project_home}").format(
            cmd=cmd, project_home=project_home
        )

    return git_branch, git_commit
