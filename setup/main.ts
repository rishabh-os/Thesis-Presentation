import { defineAppSetup } from '@slidev/types'
import VuePlyr from 'vue-plyr'
import 'vue-plyr/dist/vue-plyr.css'


export default defineAppSetup(({ app, router }) => {
  // Vue App
  app.use(VuePlyr, {
    plyr: {controls : ['play-large', 'progress'], hideControls: false,}
  })
})