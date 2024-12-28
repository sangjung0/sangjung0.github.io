FROM ruby:3.2.3-slim

LABEL Name=sangjung0github Version=0.0.1
EXPOSE 4000

# throw errors if Gemfile has been modified since Gemfile.lock
RUN bundle config --global frozen 1

RUN apt-get update && apt-get install -y git libffi-dev build-essential

# WORKDIR /workspaces/app
# COPY . /workspaces/app

# RUN bundle install --verbose

#CMD ["ruby", "sangjung0github.rb"]
#CMD bundle exec jekyll serve
CMD ["tail", "-f", "/dev/null"]
