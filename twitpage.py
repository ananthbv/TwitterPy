from bottle import get, post, request, route, run, redirect
import twitter
import urlparse
import oauth2 as oauth
import urllib
import os


cKey       = os.environ.get('TWIT_CUST_KEY', 'DoesNotExist')
cSecret    = os.environ.get('TWIT_CUST_SECRET', 'DoesNotExist')
pageipaddr = os.environ.get('TWITPAGE_IP_ADDR', 'DoesNotExist')

if cKey == 'DoesNotExist':
    print "Please set environment variable TWIT_CUST_KEY to your twitter customer key"
    exit()

if cSecret == 'DoesNotExist':
    print "Please set environment variable TWIT_CUST_SECRET to your twitter customer secret"
    exit()

if pageipaddr == 'DoesNotExist':
    print "Please set environment variable TWITPAGE_IP_ADDR to the server ip address"
    exit()

request_token_url = 'https://api.twitter.com/oauth/request_token'
access_token_url  = 'https://api.twitter.com/oauth/access_token'
authorize_url     = 'https://api.twitter.com/oauth/authorize'


@get('/login')
def login():
    return '''
        <form action="/login" method="post">
            Username: <input name="username" type="text" />
            <input value="Sign-in with Twitter" type="submit" />
        </form>
           '''

@post('/login') 
def do_login():

    username = request.forms.get('username')

    consumer = oauth.Consumer(cKey,cSecret)
    client = oauth.Client(consumer)

    resp, content = client.request(request_token_url, 
                      "POST", 
                      body='oauth_callback=http://' + pageipaddr + ':8080/authorized')
    if resp['status'] != '200':
        raise Exception("Invalid response %s." % resp['status'])

    request_token = dict(urlparse.parse_qsl(content))

    # Very bad idea to actually write to file but couldn't find a 
    # way to share variables between requests in bottle
    f = open('myfile','w')
    f.write(request_token['oauth_token_secret']) 
    f.close()

    url = 'https://api.twitter.com/oauth/authorize?%s' % (
        urllib.urlencode({
             'oauth_token'   : request_token['oauth_token'],
             'oauth_callback': 'http://' + pageipaddr + ':8080' + '/authorized'
        }).replace('+', '%20'),
    )
    url = 'https://api.twitter.com/oauth/authorize?oauth_token=' + \
          request_token['oauth_token'] + \
          '&oauth_callback=http://' + pageipaddr + ':8080/authorized'

    print "url = ", url
    return redirect(url)


@route('/authorized')
def authorized():
    consumer = oauth.Consumer(cKey,cSecret)
    f = open('myfile')
    ots = f.read()
    f.close()
    token = oauth.Token(
        request.query.oauth_token,
        ots
    )
    client = oauth.Client(consumer, token)
    res, content = client.request(
        'https://api.twitter.com/oauth/access_token',
        'POST',
        body='oauth_verifier=' + request.query.oauth_verifier
    )
    #print "content = ", content
    if res['status'] != '200':
        raise Exception("Invalid response %s: %s" % (res['status'], content))

    access_token = dict(urlparse.parse_qsl(content))

    aKey    = access_token['oauth_token']
    aSecret = access_token['oauth_token_secret']

    api = twitter.Api(consumer_key=cKey, consumer_secret=cSecret,
                      access_token_key=aKey, access_token_secret=aSecret)

    creds = api.VerifyCredentials()
    if (creds):
        print creds
        f = open('keys','w')
        f.write(aKey + "\n")
        f.write(aSecret + "\n")
        f.close()
        return '''<form action="/tweet" method="post">
               <p>Your login was successful: ''' + creds.screen_name + '''.</p> 
               <p>Enter something to tweet <p>
                  Message: <input name="message" type="text" />
            <input value="Update" type="submit" />
        </form>'''
    else:
        return "<p>Your login was unsuccessful.</p>"


@post('/tweet')
def tweet():
    message = request.forms.get('message')
    keys = [line.rstrip('\n') for line in open('keys')]
    print keys
    api = twitter.Api(consumer_key=cKey, consumer_secret=cSecret,
                      access_token_key=keys[0], access_token_secret=keys[1])

    update = api.PostUpdate(message)

    twitter_statuses = api.GetUserTimeline(screen_name = api.VerifyCredentials().screen_name)
    print len(twitter_statuses)
    statuses = []
    tweeted = False
    for t in twitter_statuses:
        if t.text == message:
            tweeted = True
            statuses.insert(0, "<p>Your tweet was successfully added</p><p>Your tweet is marked in <b>bold</b> " +
                               " in the list of last " + str(len(twitter_statuses)) + " tweets </p><p></p>")
            statuses.append("<p><b>" + t.text + "</b></p>")
        else:
            statuses.append("<p><small>" + t.text + "</small></p>")

    if tweeted:
        return statuses
    else:
        return "<p>Tweet unsuccessful</p>"


run(host='0.0.0.0', port=8080)

