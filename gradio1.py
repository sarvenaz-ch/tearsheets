# simple gradio app to access qa chain in tearsheet util
# build interface with inputs as
#   - client name (dropdown)
#   - document types to query (checkboxes)
# outputs is text from qa chain

import tearsheet_utils as tshu
import gradio
import argparse

# load vectordb - possibly make this a function in tshu - as global var
docs = tshu.load_persona_html()
vectordb = tshu.create_or_load_vectorstore('data/chroma', docs, override=False)


def get_client_names(vectordb):
    '''
    Get unique client names from vectordb metadata. 
    '''

    md = vectordb.get()['metadatas']
    return list(set([doc['client_name'] for doc in md]))


def qa(q, client_name, All, Equilar, Google, Linkedin, Pitchbook, Relsci, WealthX, ZoomInfo):
    '''
    Q&A function for app. Inputs:
        - question
        - client_name from dropdown
        - separate boolean flags for each document type
    '''

    # decide which documents to search
    if All:
        filter_docs = 'all'
    else:
        doc_types = ['Equilar', 'Google', 'Linkedin', 'Pitchbook', 'Relsci', 'WealthX', 'ZoomInfo']

        filter_docs = []  # writing as a list comprehension does not work great
        for d in doc_types:
            if eval(f'{d}==True'):
                filter_docs.append(d.lower())

    filter_ = tshu.create_filter(client_name,
        filter_docs)

    response = tshu.qa_metadata_filter(q, vectordb, filter_)

    return response


if __name__ == '__main__':
    import gradio1 as m

    # argument handler
    parser = argparse.ArgumentParser(description='Start app. Optionally share to web.')
    parser.add_argument('--share', dest='share', action='store_true',
                        help='Share the app to the web (default: False).')

    args = parser.parse_args()
    print(args)

    # get unique client names from metadata
    client_names = m.get_client_names(vectordb)

    # build gradio interface
    client_selector = gradio.Dropdown(choices=client_names, label='Client', type='value')

    # interface
    demo = gradio.Interface(
        fn=m.qa,
        inputs=["text", client_selector] + 8*["checkbox"],
        outputs=["text"],
    )
    demo.launch(share=args.share)

