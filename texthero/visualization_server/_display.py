# this file is largely based on https://github.com/jakevdp/mpld3/blob/master/mpld3/_display.py
# Copyright (c) 2013, Jake Vanderplas
# It was adapted for pyLDAvis by Ben Mabey
import warnings
import random
import json
import jinja2
import numpy
import re
import os
from ._server import serve
from .utils import get_id, write_ipynb_local_js, NumPyEncoder
import json


# General HTML template.  This should work correctly whether or not requirejs
# is defined, and whether it's embedded in a notebook or in a standalone
# HTML page.
GENERAL_HTML = jinja2.Template(
    r"""
<!DOCTYPE html>
<html lang="en">

<head>
    <link href="https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.10.16/css/jquery.dataTables.min.css" rel="stylesheet">
</head>

<body>
    <div class="container">
        <div class="header">
            <h5 class="text-muted"></h3>
        </div>

        <div>
            <div id="tablediv"></div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-1.12.4.js" type="text/javascript"></script>
    <script src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.min.js" type="text/javascript"></script>
    <script type="text/javascript">
        $(document).ready(function () {
            $("#tablediv").html({{ df_json }});
            var table = $("#tableID").DataTable();
        });

    </script>
</body>

</html>
"""
)


def prepared_data_to_html(df):
    """
    Output HTML with embedded visualization.

    Parameters
    ----------
    data : PreparedData, created using :func:`prepare`
        The data for the visualization.

    Returns
    -------
    vis_html : string
        the HTML visualization

    """
    template = GENERAL_HTML

    df_json = json.dumps(
        df.to_html(classes='table table-striped" id = "tableID', index=False, border=0),
    )

    return template.render(df_json=df_json)


def display(data, local=False, **kwargs):
    """Display visualization in IPython notebook via the HTML display hook

    Parameters
    ----------
    data : PreparedData, created using :func:`prepare`
        The data for the visualization.
    local : boolean (optional, default=False)
        if True, then copy the d3 & mpld3 libraries to a location visible to
        the notebook server, and source them from there. See Notes below.

    Returns
    -------
    vis_d3 : IPython.display.HTML object
        the IPython HTML rich display of the visualization.

    Notes
    -----
    Known issues: using ``local=True`` may not work correctly in certain cases:

    - In IPython < 2.0, ``local=True`` may fail if the current working
      directory is changed within the notebook (e.g. with the %cd command).
    - In IPython 2.0+, ``local=True`` may fail if a url prefix is added
      (e.g. by setting NotebookApp.base_url).

    """
    # import here, in case users don't have requirements installed
    from IPython.display import HTML

    return HTML(prepared_data_to_html(data))


def show(
    data,
    ip="127.0.0.1",
    port=8888,
    n_retries=50,
    local=True,
    open_browser=True,
    http_server=None,
):
    """Starts a local webserver and opens the visualization in a browser.

    Parameters
    ----------
    data : PreparedData, created using :func:`prepare`
        The data for the visualization.
    ip : string, default = '127.0.0.1'
        the ip address used for the local server
    port : int, default = 8888
        the port number to use for the local server.  If already in use,
        a nearby open port will be found (see n_retries)
    n_retries : int, default = 50
        the maximum number of ports to try when locating an empty port.
    local : bool, default = True
        if True, use the local d3 & LDAvis javascript versions, within the
        js/ folder.  If False, use the standard urls.
    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    http_server : class (optional)
        optionally specify an HTTPServer class to use for showing the
        visualization. The default is Python's basic HTTPServer.

    """

    html = prepared_data_to_html(data)

    serve(
        html,
        ip=ip,
        port=port,
        n_retries=n_retries,
        open_browser=open_browser,
        http_server=http_server,
    )
